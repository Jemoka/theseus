# Tutorial: Extend a Model and Trainer

This tutorial shows the minimal way to add a new model family and a corresponding training job.

## Step 1: Implement a Model

Create `theseus/model/models/my_model.py`:

```python
from typing import Any, Optional

from theseus.base.axis import Axis
from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.huggingface import HFCompat, LogicalAxes


class MyHFModel(HFCompat):
    id: str = field("architecture/huggingface/model")

    @property
    def sharding(self) -> list[tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value, None),
            (Axes.BLOCK_SIZE.value, None),
            (Axes.N_EMBD.value, None),
            (Axes.N_EMBD_FF.value, Axis.SHARD),
            (Axes.N_EMBD_OUT.value, Axis.SHARD),
            (Axes.N_ATTN.value, Axis.SHARD),
        ]

    @classmethod
    def axes(cls, x: str) -> Optional[LogicalAxes]:
        # map parameter name patterns to logical axes
        return None
```

If you are not using HF compat, subclass `theseus.model.module.Module` directly and implement `components()`, `sharding`, and `__call__`.

## Step 2: Register the Model Export

Update `theseus/model/models/__init__.py` so the class can be imported in experiments.

## Step 3: Implement Trainer/Experiment

Create `theseus/experiments/my_model.py`:

```python
import optax

from theseus.training.huggingface import HFTrainer, HFTrainerConfig
from theseus.evaluation.huggingface import HFEvaluator
from theseus.model.models import MyHFModel


class EvaluateMyModel(HFEvaluator[MyHFModel]):
    MODEL = MyHFModel


class PretrainMyModel(HFTrainer[MyHFModel]):
    MODEL = MyHFModel
    CONFIG = HFTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
```

For non-HF models, inherit from `BaseTrainer` instead.

## Step 4: Register Job Key

Update `theseus/experiments/__init__.py`:

```python
JOBS["my_model/train/pretrain"] = PretrainMyModel
```

## Step 5: Run It

```bash
theseus jobs | rg my_model
theseus configure my_model/train/pretrain my_model.yaml
theseus run my-model-dev my_model.yaml /tmp/theseus
```

## Design Guidance

- Keep axis mapping explicit and testable.
- Prefer minimal trainer overrides; inherit default loop behavior where possible.
- Put model-specific behavior in model/evaluator, not in generic trainer internals.
