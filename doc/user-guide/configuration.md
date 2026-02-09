# Configuration

Theseus uses dataclass-driven configuration with OmegaConf hydration.

The important idea is that config keys are **schema-backed** and **hierarchical**, not ad-hoc dictionaries.

## How Keys Are Declared

Each dataclass field uses `theseus.config.field("path/to/key", ...)` metadata.

Example pattern:

```python
from dataclasses import dataclass
from theseus.config import field

@dataclass
class MyConfig:
    lr: float = field("optimization/lr", default=3e-4)
```

## Build and Hydrate Lifecycle

1. `build(*classes)` computes canonical keys/defaults from dataclass schemas.
2. runtime enters `configuration(cfg)` context.
3. `configure(MyConfig)` hydrates strongly typed dataclass instances.

This keeps model/trainer constructors clean while preserving typed config objects.

## Slash Keys and Nested YAML

Slash keys are automatically nested into YAML sections.

`tokenizer/huggingface/use_fast` maps to:

```yaml
tokenizer:
  huggingface:
    use_fast: true
```

The reverse hydration also works for deeply nested paths.

## Multiple Config Classes

A single job can expose multiple dataclass schemas. `BaseTrainer.config()` collects model + evaluator + tokenizer + optimizer/schedule schemas.

That means one YAML can drive all subsystems while preserving defaults and type contracts.

## Override Semantics

CLI overrides are merged using OmegaConf dotlist semantics.

```bash
theseus run myrun train.yaml /tmp/theseus \
  optimization/lr=1e-4 \
  training/tokens=20000000
```

Use overrides for temporary experiments. For long-lived runs, commit the YAML.

## Common Patterns

### Pattern: base + delta configs

- keep a stable baseline config in repo,
- create variants with `--previous` + a small override set,
- store each variant with a clear filename.

### Pattern: local debugging profile

Maintain a tiny-budget profile (`tokens`, `batch_size`, `block_size`) to validate logic quickly.

### Pattern: explicit hardware hints

Populate request fields (`chip`, `min_chips`) in configs generated via CLI/Quick so promotion to remote is straightforward.

## Failure Modes to Watch

- Key typos in overrides silently creating wrong paths (inspect generated YAML).
- Hidden behavior in code instead of config schema.
- Mixing runtime-only state into config objects.
