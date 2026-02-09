# Tutorial: Quick API Iteration

This tutorial shows a practical notebook/script iteration loop.

## Scenario

You are iterating on training hyperparameters and want to inspect instantiated objects before running.

## Script

```python
from theseus.quick import quick

with quick("gpt/train/pretrain", "iter-run", "/tmp/theseus") as j:
    # Configure experiment
    j.config.training.tokens = 1_000_000
    j.config.training.batch_size = 128
    j.config.training.per_device_batch_size = 4
    j.config.architecture.block_size = 256

    # Build but do not run
    trainer = j.create()

    # Inspect runtime objects
    print("model:", type(trainer.model).__name__)
    print("state:", type(trainer.state).__name__)
    print("mesh:", trainer.mesh)

    # Run once satisfied
    trainer()
```

## Promote to YAML

```python
with quick("gpt/train/pretrain", "iter-run", "/tmp/theseus") as j:
    j.config.training.tokens = 50_000_000
    j.save("iter.yaml", chip="h100", n_chips=8)
```

Then from shell:

```bash
theseus submit iter-remote iter.yaml --chip h100 -n 8
```

## Pattern Summary

- iterate quickly in Python,
- verify object-level details,
- freeze config to YAML,
- run repeatable remote job with same settings.
