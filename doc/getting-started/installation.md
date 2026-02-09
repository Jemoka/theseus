# Installation

This page gives a practical setup path that minimizes surprises.

## Prerequisites

- Python managed by `uv`.
- Working `git` checkout of this repository.
- For GPU runs: CUDA-compatible JAX group matching your environment.

## Dependency Groups

`theseus` uses uv dependency groups in `pyproject.toml`.

General pattern:

```bash
uv sync --group <feature-group> --group <backend-group>
```

## Common Install Profiles

### CPU-only (local testing)

```bash
uv sync --group all --group cpu
```

### CUDA 13

```bash
uv sync --group all --group cuda13
```

### CUDA 12

```bash
uv sync --group all --group cuda12
```

### TPU

```bash
uv sync --group all --group tpu
```

### HuggingFace model path

Use a HF group plus backend group:

```bash
# CPU HF
uv sync --group huggingface-cpu --group cpu

# CUDA 13 HF
uv sync --group huggingface-cuda13 --group cuda13
```

## Verify Environment

```bash
uv run theseus jobs
```

You should see registered jobs such as:

- `gpt/train/pretrain`
- `llama/train/pretrain`
- data tokenization jobs

## Install Docs Tooling (optional)

```bash
uv sync --group docs
uv run mkdocs serve
```

## Next Step

Continue to `getting-started/first-run.md` for a first local run that exercises config + trainer + checkpoint paths.
