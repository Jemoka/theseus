# Adding an Experiment

An experiment is a registered job that knows which model to train and which config to use. The minimal case is three lines of actual code.

---

## Minimal experiment

```python
# theseus/experiments/models/my_model.py
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.model.models import MyModel
from theseus.registry import job


@job("my_model/train/pretrain")
class PretrainMyModel(BaseTrainer[BaseTrainerConfig, MyModel]):
    MODEL = MyModel
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls):
        return "wsd"   # warmup-stable-decay
```

Then make it importable:

```python
# theseus/experiments/__init__.py  — add one line
from .models.my_model import PretrainMyModel  # noqa: F401
```

That's it. The job now appears in `theseus jobs` under the key `my_model/train/pretrain`.

---

## The `@job` key

The string passed to `@job()` is the name used everywhere: in `theseus jobs`, in `theseus configure <key>`, and in the `job:` field in generated YAML files. Use a `/`-separated namespace to keep things organised:

```
<model>/<phase>/<variant>
gpt/train/pretrain
my_model/train/finetune
my_model/eval/perplexity
```

---

## Custom config

If your model introduces new config fields beyond what `BaseTrainerConfig` covers, create a dataclass for them and pass it to `CONFIG`:

```python
from dataclasses import dataclass
from theseus.config import field
from theseus.training.base import BaseTrainer, BaseTrainerConfig


@dataclass
class MyTrainerConfig(BaseTrainerConfig):
    # Extra fields go here.  They appear in the generated YAML automatically.
    my_loss_weight: float = field("training/my_loss_weight", default=0.1)


@job("my_model/train/pretrain")
class PretrainMyModel(BaseTrainer[MyTrainerConfig, MyModel]):
    MODEL  = MyModel
    CONFIG = MyTrainerConfig

    @classmethod
    def schedule(cls):
        return "wsd"
```

---

## Overriding training logic

`BaseTrainer` handles the full training loop. Override individual hooks if you need custom behaviour:

```python
@job("my_model/train/pretrain")
class PretrainMyModel(BaseTrainer[BaseTrainerConfig, MyModel]):
    MODEL  = MyModel
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls):
        return "wsd"

    @classmethod
    def forward(cls, state, params, batch, deterministic=False, intermediates=False):
        # Custom forward pass — same signature as BaseTrainer.forward
        ...
```

---

## Available schedules

Pass any of these strings to `schedule()`:

| Key | Description |
|---|---|
| `"wsd"` | Warmup → stable → cosine decay |
| `"wsds"` | WSD with a second stable phase for continual learning |

---

## Trying it out

```bash
# Generate a config for your new job
theseus configure my_model/train/pretrain run.yaml

# Run locally
theseus run my-run run.yaml ./output
```

See [Running Experiments](running.md) for the full workflow.
