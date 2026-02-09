# Customization Patterns

This page documents how to extend theseus without fighting the framework.

## Pattern 1: Add a New Experiment Job

### When to use

You already have a model class and want a new job key with custom schedule/evaluator.

### Approach

- subclass `BaseTrainer` for native theseus models,
- subclass `HFTrainer` for HF-compat models,
- define `MODEL`, `CONFIG`, and optional `schedule()` override,
- register the class in `theseus/experiments/__init__.py`.

### Skeleton

```python
import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation import Evaluator
from theseus.model.models import GPT


class EvaluateMyRun(Evaluator[GPT]):
    MODEL = GPT


class PretrainMyRun(BaseTrainer[BaseTrainerConfig, GPT]):
    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"

    def evaluator(self) -> Evaluator[GPT]:
        return EvaluateMyRun.from_trainer(self)
```

## Pattern 2: Add a New Model Family

### Contract

Subclass `theseus.model.module.Module` and implement:

- `components()` for config graph traversal,
- `sharding` mapping logical names to mesh axes,
- standard `__call__` forward signature.

For HF-backed models, subclass `HFCompat` and implement `axes(name)` mapping.

## Pattern 3: Custom Optimizer/Schedule

- add to `theseus/training/optimizers/` or `theseus/training/schedules/`,
- register in corresponding registry,
- return optimizer/schedule name from trainer class methods.

This keeps loop logic generic and avoids one-off trainer forks.

## Pattern 4: Dataset and Tokenization Extensions

- register dataset in `theseus/data/datasets/registry.py`,
- choose `TokenizeBlockwiseDatasetJob` or `TokenizeVariableDatasetJob`,
- for chat datasets, rely on tokenizer backend templates (`apply_chat_template` for HF, ChatML for tiktoken).

## Pattern 5: Evaluation Extension

- create evaluation subclasses under `theseus/evaluation/datasets/`,
- plug into evaluator used by experiment,
- keep inference formatting/tokenization path consistent with training inputs.

## Anti-Patterns to Avoid

- Baking model-specific behavior into `BaseTrainer` internals.
- Duplicating full trainer loops for small behavior changes.
- Hiding critical run settings in Python-only state (prefer config keys).
- Mixing logical and physical axis naming in model sharding definitions.
