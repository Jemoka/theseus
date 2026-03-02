# Running (Existing) Experiments

The typical workflow is two steps: **configure a run → run (or submit) it**.

```bash
theseus configure <job> config.yaml             # make config
```

```bash
theseus run <name> config.yaml <output_dir>     # run locally
theseus submit <name> config.yaml               # or send to remote infra
```

Before you get started, you ran run:

```bash
theseus jobs
```

This prints a table of every registered experiment. Pick one — for the examples below we'll use `gpt/pretrain`.


## Locally

### Step 1 — Generate a config

```bash
theseus configure gpt/pretrain run.yaml
```

That writes a fully-populated `run.yaml` with all default values filled in. Open it up and you'll see every knob the job exposes.

You can also 

```bash
theseus configure gpt/pretrain run.yaml training.per_device_batch_size=8 \
    model.n_layers=12
```

Planning to run on a specific chip? Bake the hardware request into the config now so you don't have to repeat it later:

```bash
theseus configure gpt/pretrain run.yaml --chip h100 -n 4
```

### Step 2 — Run it

```bash
theseus run my-gpt-run run.yaml ./output
```

- `my-gpt-run` is a human-readable name for this run (used for logging/checkpointing).
- `./output` is where checkpoints, logs, and results land.

Override config values at run time the same way:

```bash
theseus run my-gpt-run run.yaml ./output \
    training.per_device_batch_size=4
```

If you want to attach the run to a W&B project or group:

```bash
theseus run my-gpt-run run.yaml ./output \
    --project my-project --group ablation-lr
```

---

## Dispatch Remotely

Remote dispatch lets you send a job to an SSH host or a SLURM cluster from your local machine without manually SSHing in.

### Prerequisites — `~/.theseus.yaml`

You need a dispatch config that describes your infrastructure. Copy `examples/dispatch.yaml` from the repo as a starting point:

```bash
cp examples/dispatch.yaml ~/.theseus.yaml
```

Then edit it to match your clusters. A minimal plain-SSH example:

```yaml
clusters:
  mybox:
    root: /data/theseus  # this is the output folder
    work: /tmp/theseus   # this is a temporary directory whehre code is copied

hosts:
  mybox:
    ssh: mybox          # alias in ~/.ssh/config
    cluster: mybox
    type: plain
    chips:
      h100: 4
    uv_groups: [cuda12]

priority:
  - mybox
```

A minimal SLURM example:

```yaml
clusters:
  hpc:
    root: /mnt/data/theseus  # this is the output folder
    work: /scratch/theseus   # this is a temporary directory where code is copied

hosts:
  hpc-login:
    ssh: hpc            # alias in ~/.ssh/config
    cluster: hpc
    type: slurm
    partitions: [gpu]
    account: myproject
    uv_groups: [cuda12]

priority:
  - hpc-login
```

### Step 1 — Generate a config (same as before)

```bash
theseus configure gpt/pretrain run.yaml --chip h100 -n 4
```

### Step 2 — Submit

```bash
theseus submit my-gpt-run run.yaml
```

Theseus reads `~/.theseus.yaml`, finds the first host in `priority` that can satisfy the hardware request, ships your code, and either SSHs in to run it directly (plain host) or submits an `sbatch` job (SLURM host).

Override hardware at submit time if you didn't bake it into the config:

```bash
theseus submit my-gpt-run run.yaml --chip h100 -n 8
```

Pin to a specific cluster, or exclude one:

```bash
theseus submit my-gpt-run run.yaml --cluster hpc-login
theseus submit my-gpt-run run.yaml --exclude-cluster cloud
```

By default theseus ships your working tree including uncommitted changes (`--dirty`). To ship only committed code:

```bash
theseus submit my-gpt-run run.yaml --clean
```

---

## Interactive — Remote Jupyter REPL

When you want a notebook session on remote hardware instead of a batch job. First, you will need to [setup a dispatch config](#prerequisites-theseusyaml) with a host that has `uv_groups` configured for Jupyter support.

Then launch a REPL session:

```bash
theseus repl --chip h100 -n 1
```

Theseus allocates a node, starts Jupyter, and SSH-forwards the port back to `localhost:8888`. Open that URL in your browser and you're in.

### Live code sync

If your dispatch config includes a `mount` or `proxy` key, you can push local edits to a running REPL session without restarting it.

Launch with sync enabled:

```bash
theseus repl --chip h100 -n 1 --sync
```

Then, after editing files locally, push the changes:

```bash
theseus repl --update
```

Only files tracked by git are synced. Uncommitted edits are included.
