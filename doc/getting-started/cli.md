# CLI Workflow

The CLI is the most stable interface for production-style usage.

## Command Surface

- `theseus jobs`: list registered job keys.
- `theseus configure`: generate YAML from dataclass config schemas.
- `theseus run`: execute locally.
- `theseus submit`: dispatch remotely.
- `theseus checkpoints`: list available checkpoints.
- `theseus restore`: restore and continue from a checkpoint.

## Common Pattern

1. Generate config once.
2. Iterate with command-line overrides.
3. Promote exact same YAML to remote submission.

## `configure` in Practice

```bash
theseus configure gpt/train/pretrain train.yaml --chip h100 -n 8
```

Apply ad-hoc overrides:

```bash
theseus configure gpt/train/pretrain train.yaml \
  training/batch_size=1024 \
  architecture/block_size=1024
```

Merge from previous config:

```bash
theseus configure gpt/train/pretrain next.yaml --previous train.yaml \
  optimization/lr=1e-4
```

## `run` in Practice

```bash
theseus run baseline train.yaml /path/to/output
```

Temporary overrides for quick checks:

```bash
theseus run baseline train.yaml /path/to/output \
  training/tokens=200000 \
  logging/report_interval=8
```

## `submit` in Practice

```bash
theseus submit run-h100 train.yaml \
  --chip h100 \
  --chips 8 \
  --mem 128G \
  --cluster hpc-login \
  --dirty
```

Important flags:

- `--cluster`: preferred clusters (comma-separated)
- `--exclude-cluster`: forbidden clusters
- `--dirty`: include uncommitted changes in shipped payload
- `-d/--dispatch-config`: explicit path to dispatch config

## Checkpoint Commands

```bash
theseus checkpoints my-run /path/to/output
theseus restore my-run step_4096 /path/to/output
```

## Operational Advice

- Keep generated YAML committed for reproducibility.
- Avoid long override chains for real experiments; fold changes back into YAML.
- Use `restore` to validate recoverability early in project setup.
