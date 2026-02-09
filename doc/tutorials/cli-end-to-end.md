# Tutorial: CLI End to End

This tutorial walks from tokenization to local run to remote submit.

## Scenario

You want a reproducible workflow that starts local and scales remotely with no code changes.

## Step 1: Prepare Data (Example)

```bash
theseus configure data/tokenize_variable_dataset tokenize.yaml \
  data/dataset=fineweb \
  data/max_samples=200000 \
  tokenizer/backend=tiktoken \
  tokenizer/name=cl100k_base

theseus run tokenize-fineweb tokenize.yaml /tmp/theseus
```

## Step 2: Generate Training Config

```bash
theseus configure gpt/train/pretrain train.yaml \
  --chip h100 -n 8 \
  training/dataset="[{name: fineweb, rate: 1.0, style: PMD}]" \
  training/tokens=10000000 \
  architecture/block_size=512
```

## Step 3: Local Smoke Test

```bash
theseus run smoke train.yaml /tmp/theseus \
  training/tokens=500000 \
  training/batch_size=32 \
  training/per_device_batch_size=2
```

Confirm checkpoints:

```bash
theseus checkpoints smoke /tmp/theseus
```

## Step 4: Remote Submit

```bash
theseus submit prod-run train.yaml \
  --chip h100 \
  --chips 8 \
  --cluster hpc-login \
  --mem 128G \
  --dirty
```

## Step 5: Recovery Drill

```bash
theseus restore smoke step_64 /tmp/theseus
```

Do this early. Recovery issues are cheapest to find before long runs.

## Why This Pattern Works

- Same config schema for local and remote.
- CLI provides explicit, audit-friendly run history.
- Dispatch layer handles allocation, shipping, and remote bootstrap.
