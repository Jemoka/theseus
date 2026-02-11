# Dispatch and Remote Runs

This page covers both ways to launch remotely:

- CLI-driven `theseus submit`
- programmatic `theseus.dispatch.dispatch(...)`

It also explains `examples/dispatch.yaml` field-by-field.

## Dispatch Configuration File

Default path used by CLI is `~/.theseus.yaml` unless overridden via `-d`.

The canonical example is `examples/dispatch.yaml`.

## `examples/dispatch.yaml` Explained

### `priority`

Ordered list of host aliases for solver preference. Earlier entries are preferred.

### `gres_mapping`

Maps theseus chip keys to SLURM GRES names for your cluster. Example:

- `h100: h100_80gb`
- `a100-sxm4-80gb: a100_80g`

This is required when SLURM expects cluster-specific GPU type labels.

### `clusters`

Named storage/work roots used by hosts:

- `root`: canonical persistent root (data/checkpoints/results/status)
- `work`: temporary working directory for payload extraction and execution
- `log`: log output root (optional; defaults under `work`)
- `share`: shared dispatch directory for scripts (mainly for SLURM)
- `mount`: JuiceFS Redis URL (optional)
- `cache_size`, `cache_dir`: JuiceFS cache tuning

### `hosts`

Each host entry points to a cluster and chooses execution type:

- `type: plain`: direct SSH launch
- `type: slurm`: sbatch launch on scheduler

Common fields:

- `ssh`: alias from your `~/.ssh/config`
- `cluster`: cluster key from `clusters`
- `uv_groups`: dependency groups to sync on remote

Plain host specifics:

- `chips`: available chip counts for solver

SLURM host specifics:

- `partitions`
- `cpu_partitions` (optional CPU-mode partition order; fallback to `partitions` if empty)
- `account`
- `qos`
- `mem` (default memory override)
- `exclude` (nodes to avoid)
- optional `chips` limits for solver-side cap

## Submit CLI

```bash
theseus submit my-run train.yaml \
  --chip h100 \
  --n_chips 8 \
  --mem 128G \
  --cluster hpc \
  --dirty
```

### Important Submit Flags

- `--chip`, `--n_chips`: hardware request intent.
- `--cluster`: preferred cluster list.
- `--exclude-cluster`: forbidden cluster list.
- `--dirty`: include local uncommitted changes in shipped payload.
- `-d/--dispatch-config`: explicit path to dispatch YAML.

### Hardware Resolution Rules

- `chip` resolution: CLI `--chip` overrides YAML `request.chip`; otherwise YAML; otherwise `None`.
- `n_chips` resolution: CLI `--n_chips` overrides YAML `request.min_chips`; otherwise YAML; otherwise `1`.
- `n_chips == 0` forces CPU mode and clears chip selection (`chip=None`).
- `n_chips < 0` is rejected.
- If `chip` is set, it must be a supported chip key.

Modes:

- typed GPU: `chip` set and `n_chips > 0`
- generic GPU: `chip=None` and `n_chips > 0`
- CPU: `n_chips == 0`

### Solver Selection Rules

- Hosts are scanned in configured priority order (then unlisted hosts).
- Cluster allow/deny filters are applied before selection.
- First immediately satisfiable host wins.
- If no immediate match, SLURM fallback candidates are ranked by availability.

CPU mode details:

- Plain hosts are eligible (no GPU requirement).
- SLURM uses `cpu_partitions` when provided; otherwise `partitions`.
- CPU SLURM submissions omit GPU `--gres` requests.

### Runtime Hardware Finalization

For requests that serialize with `chip=None` (generic GPU and CPU mode), bootstrap now performs runtime hardware detection (same idea as `local()`), and replaces `spec.hardware` with detected chip/count/hosts while preserving dispatch cluster paths.

That means these cases can still end up with a concrete `spec.hardware.chip` and therefore a concrete `spec.topology`.

Full ASCII flowchart:

- `doc/dispatch-flow-ascii.txt`

## What Happens During Submit

`theseus submit` and `dispatch(...)` share the same backend flow:

1. validate job key exists in registry,
2. solve host allocation from request + inventory,
3. generate bootstrap payload with config/spec/hardware,
4. ship source and launch remotely,
5. for SLURM, submit packed script; for plain host, run via SSH.

## Programmatic Dispatch API

Use this when driving runs from Python orchestration code.

```python
from omegaconf import OmegaConf

from theseus.base.chip import SUPPORTED_CHIPS
from theseus.base.hardware import HardwareRequest
from theseus.base.job import JobSpec
from theseus.dispatch import dispatch, load_dispatch_config

cfg = OmegaConf.load("train.yaml")
dispatch_cfg = load_dispatch_config("~/.theseus.yaml")

result = dispatch(
    cfg=cfg,
    spec=JobSpec(name="my-run", project="proj", group="exp"),
    hardware=HardwareRequest(
        chip=SUPPORTED_CHIPS["h100"],
        min_chips=8,
        preferred_clusters=["hpc"],
    ),
    dispatch_config=dispatch_cfg,
    dirty=True,
    mem="128G",
)

print(result.ok)
```

## Operational Patterns

### Pattern: safe promotion

1. `theseus run` locally with reduced token budget.
2. keep config identical, only adjust request/scale.
3. submit remotely.

### Pattern: cluster migration

When moving clusters, usually only `~/.theseus.yaml` and `gres_mapping` need updates.

### Pattern: reproducible dispatch

Pin dispatch config in version control for team reproducibility; pass explicit `-d` in CI.
