# Adding a Model

A model is a Flax `nn.Module` that plugs into the config system. The smallest possible new model is a `GPT` subclass that overrides one thing. A fully custom model implements three methods.

---

## Option A — Extend an existing model

The fastest path: subclass `GPT` (or any other existing model) and change only what you need.

```python
# theseus/model/models/my_model.py
from theseus.model.models.base import GPT
from theseus.config import field, configure


class MyGPT(GPT):
    # Add new config-driven fields
    my_param: int = field("architecture/my_param", default=4)

    def setup(self) -> None:
        super().setup()          # keeps embedding, blocks, ln_f
        # add extra layers here

    def decode(self, x, **kwargs):
        x = super().decode(x, **kwargs)
        # post-process x
        return x
```

Then register it:

```python
# theseus/model/models/__init__.py  — add one line
from .my_model import MyGPT  # noqa: F401
```

---

## Option B — Build from scratch

Subclass `Module` directly. You must implement `setup()`, `__call__()`, `components()`, and `sharding`.

```python
# theseus/model/models/my_model.py
from typing import Any, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.model.module import Module
from theseus.model.layers import LayerNorm
from theseus.model.block import Block
from theseus.model.axes import Axes
from theseus.base.axis import Axis
from theseus.config import field, configure


class MyModel(Module):
    # Every field here is config-driven; the key maps to a YAML path.
    n_layers:   int   = field("architecture/n_layers",   default=12)
    n_embd:     int   = field("architecture/n_embd",     default=768)
    vocab_size: int   = field("architecture/vocab_size", default=100288)
    block_size: int   = field("architecture/block_size", default=512)
    dropout:    float = field("architecture/dropout",    default=0.0)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        # List every sub-module class so build() can collect their config fields.
        return [Block, LayerNorm]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value,    None),
            (Axes.N_EMBD.value,   None),
            (Axes.N_EMBD_FF.value, Axis.SHARD),
            (Axes.N_ATTN.value,   Axis.SHARD),
        ]

    def setup(self) -> None:
        self.wte = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )
        self.blocks = [configure(Block) for _ in range(self.n_layers)]
        self.ln_f   = configure(LayerNorm)

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ):
        x = jnp.take(self.wte, idx, axis=0).astype(self._activation_dtype)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, deterministic=deterministic)
        x = self.ln_f(x)
        logits = x @ self.wte.T.astype(self._activation_dtype)

        if targets is None:
            return logits, None
        loss = jnp.mean(
            nn.softmax_cross_entropy_with_integer_labels(logits[:, :-1], targets[:, 1:])
        )
        return logits, loss
```

Register it:

```python
# theseus/model/models/__init__.py
from .my_model import MyModel  # noqa: F401
```

---

## Config fields

Every `field("path/key", default=...)` on your model class will appear in the generated YAML when you run `theseus configure`. For example, `field("architecture/my_param", default=4)` maps to:

```yaml
architecture:
  my_param: 4
```

Sub-modules listed in `components()` contribute their own fields automatically — you don't need to redeclare them. See the [Config System design doc](../Design/config.md) for full details.

---

## Next step

Once your model exists, write an [experiment](adding-experiment.md) that trains it.
