import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Type, Any, Optional, Tuple

from theseus.config import field
from theseus.model.module import Module


class RMSNorm(Module):
    ndim: int = field("architecture/n_embd", default=2048)
    eps: float = field("architecture/rms_norm_eps", default=1e-6)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        weight = self.param(
            "weight", nn.initializers.ones, (self.ndim,), self._param_dtype
        )
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)
        x_norm = x_norm.astype(x.dtype)
        return (weight.astype(x.dtype)) * x_norm
