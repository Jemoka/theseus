import math
import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import List, Type, Any, Optional, Tuple

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module


class MLP(Module):
    n_embd: int = field("architecture/n_embd")
    n_layers: int = field("architecture/n_layers")
    dropout: float = field("architecture/dropout")
    bias: bool = field("architecture/bias", default=True)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    def setup(self) -> None:
        self.c_fc = nn.Dense(
            4 * self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )

        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.n_layers)),
                (Axes.N_EMBD_FF.value, Axes.N_EMBD.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)

        if not deterministic:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=False)

        return x


class QwenMLP(MLP):
    n_embd: int = field("architecture/n_embd", default=4096)
    n_layers: int = field("architecture/n_layers", default=32)
    intermediate_size: int = field("architecture/intermediate_size", default=22016)
    dropout: float = field("architecture/dropout", default=0.0)
    bias: bool = field("architecture/bias", default=False)

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)
        self.gate = nn.Dense(
            self.intermediate_size,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=init_std),
                (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.float32,
        )
        self.up = nn.Dense(
            self.intermediate_size,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=init_std),
                (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.float32,
        )
        self.down = nn.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=proj_std),
                (Axes.N_EMBD_FF.value, Axes.N_EMBD.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.float32,
        )

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        g = self.gate(x)
        u = self.up(x)
        h = jax.nn.silu(g) * u
        h = self.down(h)
        if not deterministic and self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=False)
        return h
