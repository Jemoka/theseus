# Theseus Dispatch and REPL Sync Flow

This document describes the exact runtime chain for `theseus submit`, `theseus repl`, `theseus repl --sync`, and `theseus repl --update` as currently implemented.

## Compatibility Contract

1. `submit` behavior is unchanged.
2. `bootstrap` subcommand behavior is unchanged.
3. `repl` without sync flags is unchanged.
4. New behavior is opt-in via `repl --sync` and `repl --update` only.

## Dispatch Flow (`theseus submit`)

1. CLI loads dispatch config and job config.
2. Solver chooses host/cluster (`theseus.dispatch.solve.solve_or_raise`).
3. Code is shipped with packed tar snapshot:
   - clean mode: `git archive` snapshot of `HEAD`
   - dirty mode: `git stash create` object snapshot
4. Remote bootstrap shell (`theseus/dispatch/bootstrap.sh`) is generated from template via `SlurmJob.to_bootstrap_script()`.
5. For SLURM:
   - generated bootstrap script and sbatch wrapper are copied to shared path (`cluster.share` or fallback)
   - `sbatch` submits wrapper
   - compute node runs bootstrap script
6. For plain SSH:
   - code is shipped to `cluster.work/...`
   - `_bootstrap.sh` and `_bootstrap_dispatch.py` are written remotely
   - `nohup _bootstrap.sh` starts background run

### Code shipping details

1. Dispatched payload is a packed snapshot, not a git checkout on remote.
2. Remote workdir is extracted under `cluster.work/project/group/name`.
3. This design keeps hot import paths on fast scratch/work filesystems, not shared NFS.

### JuiceFS mount details in existing dispatch

1. If `clusters.<cluster>.mount` is set, bootstrap attempts JuiceFS mount at runtime root before command execution.
2. Mount cache flags come from `cache_size` / `cache_dir`.
3. If already mounted, bootstrap does not remount.
4. Bootstrap cleanup attempts unmount on exit.

## REPL Flow (`theseus repl`)

1. CLI builds a synthetic spec: `project=repl`, `group=interactive`, `name=repl-<chip>-<timestamp>`.
2. Solver chooses target host.
3. Remote command starts Jupyter:
   - `uv run --with jupyter jupyter lab --ip 0.0.0.0 --no-browser --port 8888 --ServerApp.port_retries=50`
4. CLI polls remote log (`tail -n 200`) and parses startup URL/token.
5. On plain SSH targets, CLI opens local SSH tunnel to requested local `--port`.
6. On SLURM, CLI reports allocated hostname + notebook port.

### REPL logs

1. Plain SSH logs: `cluster.log_dir/<project>_<group>_<name>_<timestamp>.log`
2. SLURM logs: `cluster.log_dir/<project>-<group>-<name>-%j.out`
3. These are dispatch logs on cluster log directories; they are not web tracking metadata entries unless separately ingested.

## Synced REPL Launch (`theseus repl --sync`)

`--sync` keeps the same dispatch and launch path, but wraps the REPL command with a sidecar launcher.

### Sidecar injection

Only when `--sync` is set, REPL command is wrapped as:

1. export mailbox job id:
   - `THESEUS_REPL_MAILBOX_JOB_ID=${SLURM_JOB_ID:-$$}` (unless already set)
2. export workdir (`THESEUS_REPL_WORKDIR=$PWD`)
3. start sidecar:
   - `uv run python -m theseus.dispatch.mailbox.sidecar &`
4. trap EXIT/TERM/INT to kill sidecar
5. `exec uv run --with jupyter jupyter lab ...`

No sidecar is started for normal `repl` / `submit` / `bootstrap` command paths.

### Local mailbox registration after launch

After REPL launches successfully, CLI tries to register sync state:

1. Requires top-level `dispatch.yaml` field:
   - `mount: /local/mountpoint` or `proxy: [user@]host:/abs/path` (or `~/path`)
   - top-level `mount` and top-level `proxy` are mutually exclusive
2. Requires local git repo (`git rev-parse --show-toplevel`).
3. In `mount` mode, ensures local `mount` is JuiceFS:
   - if mounted and type contains `juicefs`, use as-is
   - if not mounted, mounts via target cluster backend (`clusters.<name>.mount`) using `juicefs mount -d`
   - if `juicefs` binary missing, fails registration
4. In `proxy` mode, mailbox files are accessed via existing SSH/SCP helpers (`run`, `copy_to`, `copy_from`).
5. Writes mailbox metadata under mailbox root:
   - `<mount>/mailbox`
   - or `<proxy_root>/mailbox` in proxy mode
6. Registers active job in `.active`.
7. Registers sync state quickly (records `base_rev` via `git stash create`).
8. No full baseline file-tree copy is done during launch.

If registration fails, REPL stays running; CLI prints warning.

## Mailbox Layout

Mailbox root:

- `<mount>/mailbox`
- or `<proxy_root>/mailbox` when top-level `proxy` is configured

Files/directories:

1. `.active` (JSONL): active synced REPL records
2. `jobs/<job_id>/state.json`: per-job state, backend, shadow version
3. `jobs/<job_id>/shadow/`: sender baseline snapshot (git-tracked only)
4. `jobs/<job_id>/inbox/`: pending messages
5. `jobs/<job_id>/ack/`: sidecar apply success metadata
6. `jobs/<job_id>/applied/`: archived applied messages
7. `jobs/<job_id>/failed/`: failed patch + metadata
8. `jobs/<job_id>/state/sidecar.json`: sidecar heartbeat

Message files per mail id:

1. `<mail_id>.diff` unified patch
2. `<mail_id>.meta.json` metadata
3. `<mail_id>.ready` claim sentinel

## Update Flow (`theseus repl --update`)

1. Requires local git repo.
2. Requires top-level `dispatch.yaml mount` or `proxy`.
3. Scans a single mailbox (`<mount>/mailbox/.active`) for active synced entries.
4. Applies optional `--cluster` / `--exclude-cluster` filters to those entries.
5. Prunes stale active entries before publish using sidecar heartbeat.
6. Collects backend IDs from active entries.
7. If active entries span multiple backends, aborts.
8. In mount mode, ensures local mount is active JuiceFS (auto-mount if needed).
9. In proxy mode, reads/writes mailbox via SSH/SCP to proxy root.
10. Builds current patch from git refs (`base_rev -> current stash ref`).
11. For each active job:
   - read per-job `base_rev` from `state.json` (fallback `HEAD` if invalid/missing)
   - build patch with `git diff --binary <base_rev> -- .`
   - write patch + metadata + ready sentinel to `inbox/`
   - advance job `base_rev` to current `git stash create` ref (or `HEAD`)

### Scope rules for update

1. Sync content is git-tracked files only.
2. Untracked files are excluded.
3. To sync new files, `git add` them first.
4. Binary files are skipped defensively in diff generation.

## Sidecar Runtime Behavior (`theseus.dispatch.mailbox.sidecar`)

1. Resolves root from `THESEUS_ROOT`.
2. Uses mailbox job ID from env (`THESEUS_REPL_MAILBOX_JOB_ID` / `SLURM_JOB_ID` / PID fallback).
3. Polls `jobs/<job_id>/inbox/*.ready`.
4. Claims work by atomic rename to `.processing.<pid>`.
5. Applies patch in REPL workdir:
   - `patch -p1 --forward --batch`
6. On success:
   - writes `ack/<mail_id>.ack.json`
   - archives files to `applied/`
7. On failure:
   - writes `failed/<mail_id>.fail.json`
   - archives files to `failed/`
8. Writes heartbeat file each loop.
9. Exits on SIGINT/SIGTERM; launch wrapper trap kills it on REPL shutdown.
10. Removes matching `.active` record on sidecar exit.

## Repro Checklist

### Start synced REPL

1. Ensure `dispatch.yaml` includes top-level `mount` and cluster backend mount.
2. Run:
   - `theseus repl --sync --port 8888 --chip <chip> -n <count> -d <dispatch.yaml>`
3. Verify CLI prints mailbox root + mailbox job id.

### Send updates

1. Make code change in git repo.
2. `git add` any new files you want synced.
3. Run:
   - `theseus repl --update -d <dispatch.yaml>`
4. Verify per-job `sent` statuses.

### Verify apply on target

1. Check mailbox `ack/` and `failed/` directories.
2. In notebook terminal/session, verify files updated in REPL workdir.

## Failure Cases

1. Not in git repo: sync/update fails.
2. Top-level `mount`/`proxy` missing: update fails; sync launch registration warns.
3. Mount mode: local mount exists but non-JuiceFS: fails (defensive).
4. Proxy mode: invalid proxy syntax or unreachable host/path fails with SSH/SCP error.
5. Multiple backend active entries: update fails with backend list.
6. Missing `patch` on remote host: sidecar writes failure records.
7. Dead sidecar entries are removed from `.active` during `repl --update`.
