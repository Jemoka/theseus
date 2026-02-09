# Python Quick API

The Quick API is for fast experimentation when you want code-level control without writing full launcher scripts.

## Core Object Model

`quick(...)` returns a context manager yielding `QuickJob`:

- `j.config`: mutable OmegaConf-backed configuration object.
- `j.create()`: instantiate job without running.
- `j()`: instantiate and run.
- `j.save(...)`: emit YAML with optional hardware request metadata.

## Minimal Example

```python
from theseus.quick import quick

with quick("gpt/train/pretrain", "dev-run", "/tmp/theseus") as j:
    j.config.training.per_device_batch_size = 8
    j.config.training.tokens = 500_000
    j()
```

## Create Without Running

Useful when you need to inspect model/state/data before committing to a run:

```python
with quick("llama/train/pretrain", "inspect", "/tmp/theseus") as j:
    trainer = j.create()
    print(type(trainer.model).__name__)
    print(type(trainer.state).__name__)
```

## Save Config to Promote to CLI/Dispatch

```python
with quick("gpt/train/pretrain", "export", "/tmp/theseus") as j:
    j.config.training.batch_size = 512
    j.config.optimization.lr = 2e-4
    j.save("train.yaml", chip="h100", n_chips=8)
```

Now you can run the same config via CLI:

```bash
theseus run export train.yaml /tmp/theseus
theseus submit export-remote train.yaml --chip h100 -n 8
```

## Common Usage Patterns

### Pattern: tight debug loops

- mutate `j.config` in notebook/script
- keep token budget tiny
- call `j.create()` and run custom probes

### Pattern: prototype then operationalize

- iterate with Quick API
- freeze config with `j.save`
- move execution to CLI/submit flow

## When to Prefer Quick vs CLI

Use Quick when:

- you need imperative edits inside Python,
- you want to inspect instantiated objects,
- you are prototyping architecture-level changes.

Use CLI when:

- you want auditable shell history,
- you run jobs from automation,
- you hand configs to teammates or CI.
