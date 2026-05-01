# Adding an Experiment

An experiment is a registered job that knows which model to train and which config to use. The minimal case is three lines of actual code.

---

## Minimal experiment

```python
# theseus/experiments/models/my_model.py
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.model.models import MyModel
from theseus.registry import job

# The string passed to @job() is used everywhere: in `theseus jobs`,
# in `theseus configure <key>`, and in generated YAML files.
# Convention: <model>/<phase>/<variant>  e.g. gpt/train/pretrain
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

## Available trainer base classes

Pick the base class that matches your training regime:

| Class | Import | Use for |
|---|---|---|
| `BaseTrainer` | `theseus.training.base` | From-scratch pretraining with a custom model |
| `BackbonedTrainer` | `theseus.training.backbone` | Fine-tune from a pretrained HuggingFace checkpoint; architecture and weights come from HF instead of `configure()` |
| `ContrastiveTrainer` | `theseus.training.contrastive` | DPO / preference learning with preferred and rejected pairs |
| `BackbonedContrastiveTrainer` | `theseus.training.contrastive` | DPO starting from a HuggingFace backbone |
| `KLDivergenceTrainer` | `theseus.training.kl_divergence` | Two-stage training: standard pretraining followed by pretraining with a KL penalty against the stage-1 reference policy |

`BackbonedTrainer` reads two extra config keys instead of the normal architecture section:

```yaml
architecture:
  backbone:
    implementation: llama   # "llama", "qwen", or "gpt_neox"
    weights: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

---

## Overriding initialization: `__init__(spec)`

Called once at job startup. It runs a fixed sequence of sub-steps in order:

| Method | What it does |
|---|---|
| `_init_topology(spec)` | Reads the device mesh and replica counts from `spec.topology`; computes `total_steps` from `batch_size`, `block_size`, and `total_tokens`. |
| `_init_model()` | Instantiates the model via `configure(MODEL)`, then uses `jax.eval_shape` + a JIT'd `model.init` to create sharded params directly on the right devices — no CPU materialisation. |
| `_init_state(params)` | Builds the optimizer and learning rate scheduler, then wraps params into a `TrainState` (Flax). The optimizer buffers are sharded the same way as the params. |
| `_init_batch_config(topology)` | Figures out gradient accumulation: given `batch_size` and the fitted per-device batch size, computes how many micro-batches to scan per optimizer step. |
| `_init_wandb(spec)` | Calls `wandb.init` on the main process and sets up the `Plotter` for async figure logging. |
| `_init_data(spec)` | Builds the dataset `Strategy` from the config's `datasets` list and starts async prefetch data loaders for train and val. |
| `_init_counters_and_eval()` | Resets step counter and best-score tracker; constructs the `Evaluator` by calling `self.evaluator()`. |

Override the whole `__init__` only if you need to add something after all sub-steps have run — always call `super().__init__(spec)` first:

```python
def __init__(self, spec: ExecutionSpec) -> None:
    super().__init__(spec)       # all sub-steps above have run
    self.my_aux_model = ...      # e.g. a secondary model or EMA buffer
```

More commonly, override individual sub-methods to change one specific stage without disturbing the rest. For example, `evaluator()` is the clean override point for swapping out the evaluation backend — the default returns `Evaluator.from_trainer(self)`, but you can return your own subclass.

---

## Overriding dataloading: `batch(slice="train")`

Returns the next batch from the data loader as a dict of numpy arrays with keys `x` (input token ids), `y` (target token ids), and `padding_mask`. Called once per micro-batch accumulation group.

Override it to inject extra keys, apply on-the-fly augmentation, or mix data from sources outside the normal dataset strategy:

```python
def batch(self, slice: str = "train"):
    batch = super().batch(slice)
    # add an extra signal to the batch dict
    batch["my_signal"] = compute_signal(batch["x"])
    return batch
```

The returned dict is what gets passed directly into `forward`, so any key you add here will be available there.

---

## Overriding the forward pass: `forward(state, params, batch, ...)`

A `@staticmethod` that runs one forward pass and returns `(logits, loss, meta)`. The default unpacks `batch["x"]`, `batch["y"]`, and `batch["padding_mask"]`, calls `state.apply_fn`, and returns the model's loss scalar.

Override it to compute a custom loss, add auxiliary losses, or handle a different batch layout:

```python
@staticmethod
def forward(state, params, batch, key=None, deterministic=False, intermediates=False):
    x            = batch["x"]
    y            = batch["y"]
    padding_mask = batch["padding_mask"]
    my_signal    = batch["my_signal"]    # from batch() above

    logits, loss = state.apply_fn(
        {"params": params},
        x, y,
        padding_mask=padding_mask,
        deterministic=deterministic,
    )

    aux_loss = compute_aux_loss(logits, my_signal)
    total_loss = loss + 0.1 * aux_loss

    return logits, total_loss, {}
```

`forward` is used for both the train step (called inside `jax.value_and_grad`) and the val step (called with `deterministic=True, intermediates=True`), so it must be a pure function with no side effects. The `meta` dict returned as the third element is what gets forwarded to the model's `plot()` method during validation.

---

## Available schedules

Pass any of these strings to `schedule()`:

| Key | Description |
|---|---|
| `"wsd"` | Warmup → stable → cosine decay |
| `"wsds"` | WSD with a second stable phase for continual learning |

---

## Available optimizers

Pass any of these strings to `optimizer()` (default is `"adamw"`):

| Key | Description |
|---|---|
| `"adamw"` | AdamW with global-norm gradient clipping. Config fields: `weight_decay`, `beta1`, `beta2`. |
| `"muon"` | Muon for matrix-shaped parameters (Nesterov momentum → Polar Express orthogonalisation → NorMuon variance reduction) with AdamW for embeddings, unembeddings, and scalar/bias parameters. Each group has an independent LR multiplier. |

---

## Trying it out

```bash
# Generate a config for your new job
theseus configure my_model/train/pretrain run.yaml

# Run locally
theseus run my-run run.yaml ./output
```

See [Running Experiments](running.md) for the full workflow, or [The Job System](adding-analysis-job.md) for the full hierarchy, checkpoint protocol, and how to write analysis jobs or custom restorable jobs.
