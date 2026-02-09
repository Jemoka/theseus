# Training

Training is centered on `BaseTrainer` (or `HFTrainer` for HF-compat models).

## What Trainer Owns

- model instantiation from config,
- optimizer/schedule construction,
- state sharding and mesh placement,
- data strategy initialization,
- train and validation loops,
- metric logging,
- checkpoint save/restore integration.

## Key Config Areas

- `training/*`: batch sizes, token budget, validation flags.
- `optimization/lr`
- `architecture/*`: notably `block_size`.
- `logging/*`: report/checkpoint/validation intervals.
- `training/dataset`: sampling list with dataset style and rate.

## Batch and Accumulation

The trainer computes a feasible `(per_device_batch_size, accumulate_steps)` pair. If `per_device_batch_size < 0`, it estimates batch size from memory + model size calibration.

This lets you express global batch intent while preserving per-device constraints.

## Validation Semantics

Validation inputs are reshaped into accumulation-aware shape `(S, B, T)`:

- `S`: number of micro-batches being reduced
- `B`: data-parallel batch slice
- `T`: sequence length

Loss is reduced with token-count weighting where masks are present.

## HF vs Non-HF Path

- `BaseTrainer`: native theseus module path.
- `HFTrainer`: adds buffer handling for HF compat modules, while preserving most trainer semantics.

Use `HFTrainer` when your model subclass derives from `HFCompat`.

## Practical Tuning Sequence

1. Validate shape + loss sanity on tiny token budget.
2. Stabilize batch/accumulation behavior.
3. Tune LR schedule and warmup/decay.
4. Scale hardware and enable full token budget.

## Reliability Practices

- Keep checkpoint interval short early in a project.
- Test `restore` before long runs.
- Log enough to detect NaNs quickly.
- Keep a known-good small config for regression checks.
