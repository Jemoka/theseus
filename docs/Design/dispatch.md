# Dispatch Infrastructure

This document covers every moving part of `theseus.dispatch` — from the CLI command the user types to the Python process running on a remote GPU node.

---

## Overview

Theseus can run jobs in three ways:

| Mode | Command | What happens |
|---|---|---|
| Local | `theseus run` | Job runs in-process on the current machine |
| Remote batch | `theseus submit` | Code is shipped to a remote host and a job is queued (SLURM) or launched in the background (plain SSH) |
| Remote REPL | `theseus repl` | Same ship flow, but launches Jupyter Lab and sets up a local port-forward so you can open it in your browser |

The remote flows (`submit` and `repl`) share the same pipeline: **config loading → hardware solving → code shipping → bootstrap generation → execution**.

---

## Configuration: `~/.theseus.yaml`

Everything remote is driven by a single YAML file — by default `~/.theseus.yaml`, or any path passed to `-d`.

```yaml
clusters:
  hpc:
    root: /mnt/data/theseus      # shared filesystem root (checkpoints, data)
    work: /scratch/theseus       # per-job scratch space
    log: /scratch/theseus/logs   # SLURM/SSH log files
    # Optional JuiceFS:
    mount: redis://:pw@redis.example.com:6379/0
    cache_size: 100G
    cache_dir: /scratch/juicefs-cache

hosts:
  login1:
    type: slurm
    ssh: login1                  # SSH config alias
    cluster: hpc
    partitions:
      - name: gpu
        default: true
      - name: gpu-preempt
    account: myproject
    qos: normal
    mem: 128G
    uv_groups: [cuda13]
    exclude: [bad-node-01]

  workstation:
    type: plain
    ssh: ws
    cluster: hpc
    chips:
      h100: 8
    uv_groups: [cuda12]

priority: [login1, workstation]

gres_mapping:
  h100: gpu:h100
  a100-sxm4-80gb: gpu:a100
```

**Key concepts:**

- **Clusters** define filesystem layout (`root`, `work`, `log`). Multiple hosts can share a cluster.
- **Hosts** are either `type: plain` (direct SSH) or `type: slurm` (SLURM login node). They reference a cluster by name.
- **Priority** controls which host the solver tries first.
- **`gres_mapping`** translates chip names (e.g. `h100`) to SLURM GRES strings (e.g. `gpu:h100`).

Parsed by `load_dispatch_config()` → `parse_dispatch_config()` into a `DispatchConfig` dataclass tree.

---

## Hardware Solving (`theseus/dispatch/solve.py`)

Before shipping anything, the solver finds where to run.

```
HardwareRequest  +  DispatchConfig  →  SolveResult
```

A `HardwareRequest` specifies:

- `chip` — specific chip type (e.g. `h100`), or `None` for any GPU
- `min_chips` — how many (0 = CPU-only)
- `preferred_clusters` / `forbidden_clusters` — optional cluster filters

**Solve strategy (`solve()`):**

1. Build an ordered list of hosts from `priority`, then the rest.
2. Filter by `preferred_clusters` / `forbidden_clusters`.
3. For each host in order:
   - **Plain host:** Check configured chip count. If `check_availability=True`, SSH in and run `nvidia-smi` — if any GPU has a process using ≥ 100 MiB, the host is considered busy and skipped.
   - **SLURM host:** Query partition GPU types (via `sinfo`). Filter to partitions that have the requested GRES type. If `check_availability=True`, call `squeue`/`sinfo` to count free GPUs per partition.
4. Return the first host that can satisfy `min_chips`. If none can immediately satisfy the request but SLURM hosts exist, fall back to the SLURM host with the most available GPUs and let the scheduler queue it.

`solve_or_raise()` wraps `solve()` and raises `RuntimeError` if no solution is found.

---

## Code Shipping (`theseus/dispatch/sync.py`)

Once a host is chosen, local code must reach the remote machine.

### `ship(host, remote_path)` — clean snapshot

```
git archive --format=tar.gz HEAD | ssh host "mkdir -p REMOTE && tar -xzf - -C REMOTE -m"
```

Only tracked files are included — respects `.gitignore`, excludes `.git/`.

### `ship_dirty(host, remote_path)` — include uncommitted changes

```
git stash create   # creates a stash commit without modifying the working tree
git archive --format=tar.gz <stash-commit> | ssh host "..."
```

If `git stash create` returns nothing (no changes), falls back to `HEAD`.

### `ship_files(host, remote_path, files)` — specific files only

Uses `tar -czf - -C base_path <files>` piped over SSH. Used for targeted uploads.

### `sync(host, remote_path)` — rsync incremental

Uses `rsync -az` with standard exclusions (`.git`, `__pycache__`, `.venv`, `*.pt`, etc.). More efficient for repeated syncs.

---

## Bootstrap Script Generation (`theseus/dispatch/dispatch.py`)

The key insight: config, hardware allocation result, and job identity are all **embedded directly into the bootstrap Python script** at dispatch time. No config files need to be transferred separately.

`_generate_bootstrap(cfg, hardware, spec)` reads `theseus/dispatch/bootstrap.py` as a template and substitutes four placeholders:

| Placeholder | Value |
|---|---|
| `__CONFIG_YAML__` | Full OmegaConf YAML of the job config |
| `__HARDWARE_JSON__` | JSON-serialized `HardwareResult` (chip, hosts, cluster paths) |
| `__JOB_NAME__` | The run name |
| `__PROJECT__` / `__GROUP__` | Organizational metadata |

The resulting `_bootstrap_dispatch.py` is self-contained and can be run with `python _bootstrap_dispatch.py` from anywhere inside the shipped repo.

Similarly, `bootstrap.sh` (the shell wrapper) is generated from `theseus/dispatch/bootstrap.sh` with placeholders filled in by `SlurmJob.to_script()` — including `__WORKDIR__`, `__PAYLOAD_EXTRACT__`, `__UV_SYNC__`, `__COMMAND__`, `__JUICEFS_MOUNT__`, etc.

---

## Flow A: Plain SSH Dispatch (`_dispatch_plain`)

Used when the solved host is `type: plain`.

```
Local                           Remote (SSH alias)
─────────────────────────────   ──────────────────────────────────────────────
1. ship() or ship_dirty()   →   mkdir -p WORK_DIR && tar -xzf - -C WORK_DIR
2. write _bootstrap.sh      →   cat > WORK_DIR/_bootstrap.sh
3. write _bootstrap_dispatch.py →  cat > WORK_DIR/_bootstrap_dispatch.py
4. launch                   →   mkdir -p LOG_DIR
                                chmod +x _bootstrap.sh
                                nohup _bootstrap.sh > LOG_FILE 2>&1 &
                                (returns immediately)
Returns RunResult with log path
```

The job runs in the background; the CLI returns once the `nohup` command succeeds. Logs are at `LOG_DIR/{project}_{group}_{name}_{timestamp}.log`.

**Inside `_bootstrap.sh` on the remote:**

1. Sources `~/.bashrc`.
2. Installs `uv` if missing (via the official install script).
3. Installs `juicefs` if missing (downloads from GitHub releases).
4. Mounts JuiceFS at `cluster.root` if `mount:` is configured.
5. Loads environment modules and sets env vars.
6. Extracts the shipped tarball into `WORK_DIR` (via `__PAYLOAD_EXTRACT__`).
7. `cd WORK_DIR`
8. Runs `uv python install 3.11` then `uv sync --group <uv_groups>`.
9. Runs the main command: `python _bootstrap_dispatch.py`.
10. On exit (success, failure, or signal), cleanup handler: unmounts JuiceFS, removes work dir on success, preserves it on failure.

---

## Flow B: SLURM Dispatch (`_dispatch_slurm`)

Used when the solved host is `type: slurm`.

The key difference: the payload (code tarball + bootstrap scripts) is uploaded to a **shared directory** visible to both the login node and compute nodes, then `sbatch` is submitted from the login node.

```
Local                        Login Node (SSH)            Compute Node(s)
────────────────────────────  ─────────────────────────  ───────────────────────
1. ship() or ship_dirty()  →  share_dir/<job>.tar.gz
2. _bootstrap.sh           →  share_dir/_bootstrap.sh
3. _bootstrap_dispatch.py  →  share_dir/_bootstrap_dispatch.py
4. sbatch.sh               →  share_dir/sbatch.sh
5. sbatch share_dir/sbatch.sh (login node queues job)

                              [SLURM allocates nodes]

                                                          srun bash -l _bootstrap.sh
                                                          (runs on each allocated node)
                                                          → installs uv/juicefs
                                                          → mounts JuiceFS
                                                          → extracts payload to WORK_DIR
                                                          → uv sync
                                                          → python _bootstrap_dispatch.py
```

`submit_packed()` in `slurm.py`:

1. Calls `ship()` or `ship_dirty()` to get a tarball.
2. SCP's the tarball, `bootstrap.sh`, `_bootstrap_dispatch.py`, and `sbatch.sh` to `share_dir/`.
3. SSH's to login node and runs `sbatch share_dir/sbatch.sh`.
4. Parses the SLURM job ID from sbatch output.
5. Returns `SlurmResult(ok=True, job_id=...)`.

**`sbatch.sh`** sets SBATCH directives (`--nodes`, `--gres`, `--partition`, `--account`, `--mem`, `--time`, `--output`, etc.) then calls:

```bash
srun --wait=120 bash -l share_dir/_bootstrap.sh
```

`srun` runs `bootstrap.sh` on every allocated node simultaneously.

**Inside `bootstrap.sh` on compute nodes**, `__PAYLOAD_EXTRACT__` expands to:

```bash
mkdir -p WORK_DIR
tar -xzf share_dir/<job>.tar.gz -C WORK_DIR -m
```

So all nodes share the same tarball but each extracts to their own scratch `WORK_DIR`.

---

## Bootstrap Python Execution (`theseus/dispatch/bootstrap.py`)

Once `bootstrap.sh` has set up the environment and installed dependencies, it runs:

```bash
python _bootstrap_dispatch.py
```

This is the filled-in `bootstrap.py` template. It:

1. Loads the embedded `CONFIG_YAML` string via `OmegaConf.create()`.
2. Applies runtime config overrides from environment variables:
   - `THESEUS_DISPATCH_BATCH_SIZE` — override `training.per_device_batch_size`
   - `THESEUS_DISPATCH_DISABLE_WANDB` — set `logging.wandb = False`
3. Reconstructs `HardwareResult` from `HARDWARE_JSON`.
   - If `chip` was `None` (generic GPU request), runs `local()` to detect actual devices.
   - Supports `THESEUS_DISPATCH_ROOT_OVERRIDE` / `WORK_OVERRIDE` / `LOG_OVERRIDE` env vars.
4. Builds `Topology` (device mesh, tensor parallelism sharding).
5. Creates `ExecutionSpec(name, project, group, hardware, topology, distributed)`.
6. Sets up status directory at `cluster.root/status/{project}/{group}/{name}/{run_id}/`.
7. Generates a unique `run_id` from timestamp + `SLURM_JOB_ID` if available.
8. Writes `metadata.json` with status `"running"`.
9. Starts a `HeartbeatUpdater` thread (updates `metadata.json` every 30 s).
10. Registers SIGTERM/SIGINT handlers to mark status as `"preempted"` on SLURM preemption.
11. Looks up `cfg.job` in `JOBS` registry to get the job class.
12. **Idempotent dispatch:** If the job class is a `RestoreableJob`, checks for an existing checkpoint via `RestoreableJob.latest(spec)`. If found, restores from it instead of starting fresh.
13. Runs the job inside `with configuration(cfg):` → `job_cls(spec)()`.
14. On completion: stops heartbeat, writes `metadata.json` with status `"completed"`.
15. On exception: writes status `"failed"`, re-raises.

---

## Auto Batch-Size Search

`bootstrap.sh` contains a feature for automatically finding the largest per-device batch size that fits in GPU memory — triggered when `per_device_batch_size: -1` appears in `_bootstrap_dispatch.py`.

When triggered, `bootstrap.sh` spawns an inline Python program that:

1. Iterates over a candidate list: `[1024, 512, 256, 128, 64, 32, 16, 10, 8, 4, 3, 2, 1]`.
2. For each candidate, launches a probe subprocess with `THESEUS_DISPATCH_BATCH_SIZE=N` and `THESEUS_DISPATCH_DISABLE_WANDB=1`.
3. Watches stdout for `RESOURCE_EXHAUSTED` or `JaxRuntimeError: INTERNAL:` markers.
4. Binary-searches between the last OOM and the last successful batch size.
5. Once the best size is found, launches the real run (with wandb re-enabled) at that batch size.

Probes have adaptive timeouts based on how long previous OOM runs took, with cooldown periods between allocations to let GPU memory settle.

---

## Flow C: REPL Dispatch (`dispatch_repl`)

REPL dispatch follows the same solve + ship flow but instead of running `python _bootstrap_dispatch.py`, the bootstrap script's `__COMMAND__` is set to launch Jupyter Lab:

```bash
uv run --with jupyter jupyter lab --ip 0.0.0.0 --no-browser --port 8888 --ServerApp.port_retries=50
```

After launching, `dispatch_repl` polls the remote log file (via repeated `tail -n 200 log_file` over SSH) waiting for a URL like:

```
http://0.0.0.0:8888/lab?token=abcdef123
```

Once found, it resolves the port and establishes a local SSH tunnel:

```bash
ssh -N -L LOCAL_PORT:localhost:8888 SSH_ALIAS
```

Returns a `ReplResult` with `local_url = "http://localhost:LOCAL_PORT/lab?token=..."`.

### REPL on SLURM

On SLURM the flow is:

1. `submit_packed()` → `sbatch` → SLURM allocates and runs on a compute node.
2. `wait_until_running(job_id, ssh_alias)` — polls `squeue` until the job transitions to `R` state. Returns the allocated hostname.
3. Log is at `{log_dir}/{job_name}-{job_id}.out` — poll this for Jupyter URL.
4. SSH tunnel goes through the login node to the compute node's `localhost:8888`.

If interrupted while waiting, `cancel(job_id, ...)` is called to avoid orphaned allocations.

### REPL Sync (Mailbox)

When `--sync` is passed, the REPL runs with an extra sidecar process and the local client can push code changes live.

**Remote side** (inside the bootstrap command):

```bash
uv run --with jupyter jupyter lab ... &
NOTEBOOK_PID=$!
uv run python -m theseus.dispatch.mailbox.sidecar &
SIDECAR_PID=$!
wait $NOTEBOOK_PID
```

The sidecar (`mailbox/sidecar.py`) watches a per-job `inbox/` directory for `.ready` files and applies incoming git patches to the working directory.

**Local side** (`theseus sync` / `publish_updates()`):

1. Reads `~/.theseus.yaml` for `mount:` (local JuiceFS mount path) or `proxy:` (SSH proxy host).
2. Looks up active REPL sessions from `{mount}/mailbox/.active`.
3. For each active session, computes `git diff base_rev -- .` to get a patch.
4. Writes `{inbox}/{mail_id}.diff` + `{mail_id}.meta.json` + `{mail_id}.ready` atomically.
5. Updates `state.json` so the next sync knows the new base revision.

Transport options:
- **JuiceFS mount:** local and remote share the same filesystem; files written locally appear on the remote immediately.
- **SSH proxy:** files are SCP'd to a remote server that both the local machine and the compute node can reach.

---

## Data Flow Summary

```
theseus submit my-run train.yaml
         │
         ▼
   cli.py: submit()
         │  load config (OmegaConf)
         │  load dispatch config (~/.theseus.yaml)
         │  build HardwareRequest
         │  build JobSpec
         ▼
   dispatch.dispatch()
         │  validate job in JOBS registry
         ▼
   solve.solve_or_raise()
         │  check hosts in priority order
         │  (optional) check GPU availability via SSH/squeue
         │  → SolveResult(host_name, is_slurm, partition, hardware)
         ▼
   _generate_bootstrap()
         │  fill bootstrap.py template with CONFIG_YAML + HARDWARE_JSON
         ▼
         ├── is_slurm=True  →  _dispatch_slurm()
         │                        slurm.submit_packed()
         │                          sync.ship() / ship_dirty()
         │                          scp bootstrap files to share_dir
         │                          ssh: sbatch sbatch.sh
         │                          → SlurmResult(job_id)
         │
         └── is_slurm=False →  _dispatch_plain()
                                  sync.ship() / ship_dirty()
                                  ssh: cat > _bootstrap.sh
                                  ssh: cat > _bootstrap_dispatch.py
                                  ssh: nohup _bootstrap.sh > log 2>&1 &
                                  → RunResult(log_path)

                                     [on remote node]
                                  bootstrap.sh:
                                    ensure_uv / ensure_juicefs
                                    mount JuiceFS (if configured)
                                    extract payload to WORK_DIR
                                    uv sync
                                    python _bootstrap_dispatch.py
                                      load CONFIG_YAML + HARDWARE_JSON
                                      check for checkpoint (idempotent)
                                      with configuration(cfg):
                                          job_cls(spec)()
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `theseus/cli.py` | User-facing `theseus submit / run / repl / sync` commands |
| `theseus/dispatch/config.py` | Parse `~/.theseus.yaml` into `DispatchConfig`; `RemoteInventory` lookups; `discover_plain_host` / `discover_slurm_partitions` |
| `theseus/dispatch/solve.py` | `solve()` — pick a host and allocation given a `HardwareRequest` |
| `theseus/dispatch/sync.py` | `ship()`, `ship_dirty()`, `ship_files()`, `sync()` — move code to remote |
| `theseus/dispatch/ssh.py` | Low-level SSH / SCP wrappers with rate-limiting, backoff, retries; `forward_port()` for SSH tunnels |
| `theseus/dispatch/slurm.py` | `SlurmJob`, `submit_packed()`, `status()`, `cancel()`, `wait_until_running()`, `available_gpus()`, etc. |
| `theseus/dispatch/dispatch.py` | Top-level `dispatch()` and `dispatch_repl()` orchestrators; bootstrap script generation |
| `theseus/dispatch/bootstrap.py` | Template that runs on the remote node: load embedded config, reconstruct hardware, run job |
| `theseus/dispatch/bootstrap.sh` | Shell wrapper: install uv/juicefs, mount JuiceFS, extract payload, run Python |
| `theseus/dispatch/sbatch.sh` | SLURM batch script template: SBATCH directives + `srun bootstrap.sh` |
| `theseus/dispatch/mailbox/mailbox.py` | Mailbox protocol: register synced REPL sessions, publish code patches |
| `theseus/dispatch/mailbox/sidecar.py` | Remote sidecar: watch inbox, apply patches to live REPL working directory |
