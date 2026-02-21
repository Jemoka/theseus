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
    intermediate_size: int = field("architecture/intermediate_size", default=-1)
    bias: bool = field("architecture/bias", default=True)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    @property
    def _compute_dtype(self) -> Any:
        return jnp.bfloat16

    @property
    def _intermediate_features(self) -> int:
        return self.intermediate_size if self.intermediate_size > 0 else 4 * self.n_embd

    def _make_dense(
        self,
        features: int,
        sharding_axes: Tuple[str, str],
        stddev: float,
    ) -> nn.Dense:
        return nn.Dense(
            features,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=stddev),
                sharding_axes,
            ),
            param_dtype=jnp.float32,
            dtype=self._compute_dtype,
        )

    def setup(self) -> None:
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)
        self.c_fc = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            init_std,
        )
        self.c_proj = self._make_dense(
            self.n_embd,
            (Axes.N_EMBD_FF.value, Axes.N_EMBD.value),
            proj_std,
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
    def _compute_dtype(self) -> Any:
        return jnp.float32

    def setup(self) -> None:
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)
        self.gate = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            init_std,
        )
        self.up = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            init_std,
        )
        self.down = self._make_dense(
            self.n_embd,
            (Axes.N_EMBD_FF.value, Axes.N_EMBD.value),
            proj_std,
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


class NeoXMLP(MLP):
    n_embd: int = field("architecture/n_embd", default=2048)
    n_layers: int = field("architecture/n_layers", default=24)
    intermediate_size: int = field("architecture/intermediate_size", default=8192)
    dropout: float = field("architecture/dropout", default=0.0)
    hidden_act: str = field("architecture/hidden_act", default="gelu")
    bias: bool = field("architecture/bias", default=True)

    @property
    def _compute_dtype(self) -> Any:
        return jnp.float32

    def setup(self) -> None:
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)
        self.dense_h_to_4h = self._make_dense(
            self._intermediate_features,
            (Axes.N_EMBD.value, Axes.N_EMBD_FF.value),
            init_std,
        )
        self.dense_4h_to_h = self._make_dense(
            self.n_embd,
            (Axes.N_EMBD_FF.value, Axes.N_EMBD.value),
            proj_std,
        )

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        x = self.dense_h_to_4h(x)
        if self.hidden_act == "gelu_new":
            x = jax.nn.gelu(x, approximate=True)
        elif self.hidden_act == "gelu":
            # Manual erf-based GELU: avoids XLA pattern-matching jax.nn.gelu
            # into a less precise fused kernel, giving ~100x better parity with PyTorch
            x = x * (jax.lax.erf(x / jnp.sqrt(2.0)) + 1) / 2
        else:
            x = getattr(jax.nn, self.hidden_act, jax.nn.gelu)(x)
        x = self.dense_4h_to_h(x)
        if not deterministic and self.dropout > 0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=False)
        return x


class LlamaMLP(QwenMLP):
    n_embd: int = field("architecture/n_embd", default=4096)
    n_layers: int = field("architecture/n_layers", default=32)
    intermediate_size: int = field("architecture/intermediate_size", default=11008)
    dropout: float = field("architecture/dropout", default=0.0)
    bias: bool = field("architecture/bias", default=False)

    # Inherits setup/forward from QwenMLP; shapes and activation (SiLU) match Llama
