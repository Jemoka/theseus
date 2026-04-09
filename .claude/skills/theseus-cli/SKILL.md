---
name: theseus-cli
description: "Manage the full lifecycle of theseus ML training jobs: configure YAML configs, run jobs locally, submit/dispatch to remote clusters (SLURM, SSH, TPU, Volcano), start remote Jupyter REPLs, generate bootstrap scripts, and manage checkpoints. Use this skill whenever the user wants to train a model, launch a job, edit a training config, dispatch to a cluster, check on checkpoints, set up a remote notebook, or do anything involving the `theseus` CLI — even if they don't say 'theseus' explicitly. Also trigger when the user references configs in configs/, mentions job names like 'gpt/train/pretrain' or 'continual/train/abcd', or talks about chips/GPUs/TPUs in the context of running experiments."
---

# Theseus CLI Skill

You are helping the user work with the `theseus` CLI — the command-line interface for configuring, running, and dispatching ML training jobs in this repository.

## Running commands

This project uses **uv** as its package manager. Always invoke the CLI through `uv run`:

```
uv run theseus <command> [options...]
```

Never call `theseus` directly — it may not be on PATH or may resolve to a different environment. `uv run` ensures the correct virtualenv and dependencies.

For dependency setup:
- CUDA 13: `uv sync --group all --group cuda13`
- CUDA 12: `uv sync --group all --group cuda12`
- TPU: `uv sync --group all --group tpu`
- CPU: `uv sync --group all --group cpu`

All commands accept `-v` / `--verbose` for debug logging.

## Quick reference

| Command | What it does |
|---------|-------------|
| `theseus configure` | Generate a YAML config for a registered job |
| `theseus run` | Run a job locally |
| `theseus submit` | Dispatch a job to remote infrastructure |
| `theseus repl` | Start a remote Jupyter REPL on a cluster |
| `theseus bootstrap` | Generate a self-contained bootstrap shell script |
| `theseus checkpoints` | List checkpoints for a job |
| `theseus restore` | Restore and continue from a checkpoint |
| `theseus jobs` | List all registered jobs |

## Understanding the config system

Configs are YAML files with hierarchical keys that map to Python dataclass fields. The standard top-level sections are:

- `architecture/` — model shape (n_layers, n_embd, n_head, dtype, block_size, etc.)
- `training/` — batch_size, tokens, dataset list, validation
- `optimization/` — lr, weight_decay, betas, warmup/decay
- `eval/` — which evaluations to run
- `tokenizer/` — backend (tiktoken/huggingface), name
- `logging/` — intervals, wandb, plots
- `job` — registered job name (e.g. `"gpt/train/pretrain"`)
- `request/` — hardware: chip, min_chips, n_shards

Existing configs live in `configs/` organized by experiment family:
- `configs/gpt/` — GPT pretraining variants (small, big, moe, dict, hardware-specific)
- `configs/continual/` — continual learning experiments (safety, domain_shift, benchmarks, etc.)
- `configs/data/` — dataset-specific configs by tokenizer (cl100k, llama)
- `configs/forking/` — model forking experiments
- `configs/scratch/` — scratch/test configs

When editing or creating configs, always respect the existing directory structure. Place new configs alongside related ones.

### Multi-phase training

Continual learning configs support multi-phase training. Tokens and datasets become lists-of-lists:

```yaml
training:
  tokens: [20000000000, 2000000000]
  dataset:
    - - {name: fineweb, rate: 1.0, style: PMD}
    - - {name: mmlu, rate: 0.5}
      - {name: squad, rate: 0.5}
  fade:
    overlap: 0.0
    curve: linear
```

### Hardware request

The `request` block tells the dispatcher what hardware to target:

```yaml
request:
  chip: h100       # see supported chips below
  min_chips: 4     # 0 means CPU-only
  n_shards: 1      # tensor parallel shards
```

**Supported chips:** cpu, gb10, b200, h200, h100, a100-sxm4-80gb, a100-pcie-40gb, a6000, ada6000, l40, l40s, drive-pg199, tpu-v2, tpu-v3, tpu-v4, tpu-v5e, tpu-v5p

## Command details

### `theseus configure`

Generates a config YAML from a job's default schema, optionally merging a previous config and applying overrides.

```
theseus configure <JOB> <OUT_YAML> [OPTIONS] [OVERRIDES...]
```

Options:
- `-p / --previous <yaml>` — base config to merge onto the new schema
- `--chip <name>` — chip type for request block
- `-n / --n_chips <int>` — chip count
- `--n_shards <int>` — tensor parallel shards
- `--load <module>` — preload Python modules (repeatable)

Overrides are `key=value` pairs using dot notation: `training.batch_size=256 optimization.lr=0.001`

### `theseus run`

Runs a job locally.

```
theseus run <NAME> <YAML_PATH> <OUT_PATH> [OPTIONS] [OVERRIDES...]
```

Options:
- `-j / --job <name>` — override job name (default: read from YAML)
- `-p / --project <name>` — W&B project
- `-g / --group <name>` — W&B group
- `-s / --stage <yaml>` — additional stage configs for multi-stage pipelines (repeatable)
- `--restore <path>` — restore from checkpoint
- `--load <module>` — preload modules (repeatable)

### `theseus submit`

Dispatches a job to remote infrastructure. Requires a dispatch config (`~/.theseus.yaml` or `-d`).

```
theseus submit <NAME> <YAML_PATH> [OPTIONS] [OVERRIDES...]
```

Key options:
- `-d / --dispatch-config <path>` — dispatch config (default: `~/.theseus.yaml`)
- `-j / --job`, `-p / --project`, `-g / --group` — job metadata
- `--chip`, `-n / --n_chips`, `--n_shards` — hardware
- `--mem <size>` — memory (e.g. "64G")
- `--cluster <names>` — restrict to clusters (comma-separated)
- `--exclude-cluster <names>` — exclude clusters
- `--dirty / --clean` — include uncommitted changes (default: dirty)
- `--target <group>` — extra uv dependency groups (repeatable)
- `-s / --stage <yaml>` — multi-stage pipeline stages (repeatable)
- `--restore <path>` — restore from checkpoint
- `--load <module>` — preload modules (repeatable)
- TPU overrides: `--tpu-version`, `--tpu-spot/--tpu-on-demand`, `--tpu-preemptible/--tpu-no-preemptible`
- Volcano overrides: `--volcano-image`, `--volcano-namespace`

### `theseus repl`

Starts a remote Jupyter notebook on cluster hardware.

```
theseus repl [OPTIONS]
```

Key options:
- Hardware and dispatch options (same as submit)
- `--sync` — enable mailbox sync sidecar
- `--update` — send patches to active synced REPLs
- `--port <int>` — local port (default: 8888)
- `--startup-timeout <secs>` — wait for Jupyter (default: 180)

### `theseus bootstrap`

Generates a standalone shell script that unpacks and runs a job anywhere.

```
theseus bootstrap <NAME> <YAML_PATH> <OUT_SCRIPT> [OPTIONS] [OVERRIDES...]
```

Key options:
- `--root`, `--work`, `--log` — cluster paths (root can be set at runtime)
- `--mount <redis-url>` — JuiceFS mount
- `--cache-size`, `--cache-dir` — JuiceFS cache
- `--target <group>` — uv dependency groups (repeatable)
- `--dirty / --clean`

### `theseus checkpoints` / `theseus restore`

```
theseus checkpoints <NAME> <OUT_PATH> [-p PROJECT] [-g GROUP]
theseus restore <NAME> <CHECKPOINT> <OUT_PATH> [-p PROJECT] [-g GROUP]
```

## Output paths

The output path for `theseus run` should be the **root directory** (e.g. `~/theseus`), not a nested per-job path. The job itself creates subdirectories under `<root>/data/<dataset_name>/` etc. If you point output to `~/theseus/data/mydata`, you'll get double-nesting like `~/theseus/data/mydata/data/mydata/`.

For tokenization jobs: `uv run theseus run <name> <config> ~/theseus`
The output lands at `~/theseus/data/<dataset_name>/train.bin` etc.

## TPU dispatch

- TPU dispatch does **not** support `per_device_batch_size=-1` (AUTO_BATCH). You must set an explicit `per_device_batch_size` via config or override.
- TPU pods typically need `n_shards > 1` for tensor parallelism (e.g. `--n_shards 4` for v4-32).
- Pass per_device_batch_size as a CLI override: `training.per_device_batch_size=4`

## How to help the user

### When the user wants to create or edit a config

1. **Always search first.** Before creating anything, look through `configs/` for an existing config that matches what the user needs. Use glob and grep to search by job name, dataset, chip, or keywords. There are many configs already — the right one probably exists or is close enough to adapt.
2. If a close match exists, read it, propose edits or overrides, and use it as-is or as a `-p / --previous` base.
3. Only create a new config if nothing suitable exists. New configs go in `configs/` under the appropriate subdirectory (e.g. `configs/gpt/`, `configs/continual/`). Use `theseus configure` to generate a skeleton from the job's schema, then edit as needed.
4. **Always use `theseus configure`** to generate new configs rather than writing YAML by hand. Use `-p` to base on an existing config and overrides to change fields.

### When the user wants to run something locally

Construct a `theseus run` command. Make sure:
- The YAML exists and has a `job` key (or use `-j`)
- The output path is the **root directory** (e.g. `~/theseus`), not a per-job subdirectory
- Any overrides are in `key=value` dot notation

### When the user wants to dispatch to a cluster

1. Check that `~/.theseus.yaml` exists (or ask for dispatch config path)
2. Verify the config YAML has a `request` block with appropriate chip/min_chips, or pass them via CLI flags
3. Construct the `theseus submit` command with all needed options
4. Remind about `--dirty` (default) vs `--clean` if relevant

### When the user mentions a job name

Job names are slash-separated paths like `gpt/train/pretrain` or `continual/train/abcd`. Use `theseus jobs --load <module>` if the user has custom jobs defined outside the built-in registry.

### When constructing commands

Always show the full command to the user before running it. These commands can launch real compute jobs that cost money — never run `theseus submit`, `theseus repl`, or `theseus bootstrap` without the user seeing and confirming the command first.

### Dispatching sweeps

When dispatching many related jobs (e.g. a sweep over a parameter):

1. **Generate configs with `theseus configure`** in a loop, using `-p` to base on an existing config and overrides for the swept parameter. Name configs systematically (e.g. `dict_v32.yaml`, `dict_v128.yaml`).
2. **Tokenize data first** if the sweep introduces new dataset variants. Run tokenization locally with `theseus run` pointing to the root output dir.
3. **Dispatch in parallel** using shell backgrounding (`&` + `wait`). Group by model variant × sweep parameter.
4. **Keep naming consistent**: job names like `{model}_dict_v{N}`, configs in the appropriate subdirectory (`configs/gpt/`, `configs/scratch/`, `configs/forking/`).
5. When sweeping a parameter that affects vocab_size (like n_values in dictlearn), make sure `architecture.vocab_size` in the config matches the dataset's actual vocab size.

### Common hardware presets

| Use case | Chip | Count | Memory | Notes |
|----------|------|-------|--------|-------|
| Small/dict experiments | a6000 | 1 | 32G | Default for quick runs |
| Big GPU runs | a100-sxm4-80gb | 4 | 128G | For ~1.8B models |
| Big TPU runs | tpu-v4 | 32 | 32G | Needs explicit per_device_batch_size and n_shards=4 |
