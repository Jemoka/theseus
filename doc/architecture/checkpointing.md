# Checkpointing and Recovery

Checkpoint handling is implemented in `theseus/job.py`.

## Stored Artifacts

For each checkpoint suffix:

- model/training tree via Orbax (`checkpoint/`),
- random states (`rng.npy`),
- metadata (`config.json`),
- job spec (`job.json`),
- full YAML config (`config.yaml`).

Plus `latest` pointer for convenient resume.

## Multi-Host Safety

Save/restore uses global host barriers and global array handling to avoid inconsistent writes.

## Restore Flow

`RestoreableJob.from_checkpoint(...)`:

1. loads saved job spec and config,
2. instantiates the appropriate registered job class,
3. calls job-specific `restore(suffix)` implementation.

## CLI Recovery

Use:

```bash
theseus checkpoints <name> <out_path>
theseus restore <name> <checkpoint_suffix> <out_path>
```
