# First Local Run

This is the fastest path to verify your stack end-to-end.

## Goal

Run a minimal local training job with a generated config and confirm that:

- config hydration works,
- trainer loop starts,
- checkpoints/metrics paths are writable.

## 1. Create Output Directory

```bash
mkdir -p /tmp/theseus-demo
```

## 2. Generate a Training Config

```bash
theseus configure gpt/train/pretrain /tmp/theseus-demo/train.yaml \
  --chip cpu -n 1 \
  training/tokens=200000 \
  training/batch_size=32 \
  training/per_device_batch_size=2 \
  architecture/block_size=128 \
  logging/report_interval=8 \
  logging/validation_interval=64 \
  logging/checkpoint_interval=64
```

Notes:

- For local bring-up, keep token budget and block size small.
- `--chip` and `-n` annotate request info in the config.

## 3. Run the Job

```bash
theseus run first-local /tmp/theseus-demo/train.yaml /tmp/theseus-demo
```

## 4. Check Progress

```bash
theseus checkpoints first-local /tmp/theseus-demo
```

You should see one or more checkpoint suffixes when checkpoint interval is reached.

## 5. Restore Test

```bash
theseus restore first-local <checkpoint-suffix> /tmp/theseus-demo
```

If restore succeeds, your local lifecycle is healthy.

## Troubleshooting

### No jobs listed

- Verify dependencies were synced (`uv sync ...`).
- Run `uv run theseus jobs` inside the repo environment.

### OOM on local

- Reduce `architecture/block_size`.
- Reduce `training/per_device_batch_size`.
- Use CPU profile for pure functional testing.

### Validation shape errors

- Ensure your dataset tokenization output actually contains enough tokens for the configured block size and validation steps.

## Next Step

- Read `getting-started/cli.md` for full command semantics.
- Or jump to `tutorials/cli-end-to-end.md` for a realistic workflow.
