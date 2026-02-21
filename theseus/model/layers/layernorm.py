import jax
import flax.linen as nn
import jax.numpy as jnp

from typing import List, Type, Any

from theseus.config import field
from theseus.model.module import Module


class LayerNorm(Module):
    ndim: int = field("architecture/n_embd")
    bias: bool = field("architecture/bias")
    # Use float32 for parity with HF GPT-NeoX/LLaMA style norms
    dtype: jnp.dtype = jnp.float32
    eps: float = field("architecture/layer_norm_eps", default=1e-5)

    @property
    def sharding(self) -> list[tuple[str, Any | None]]:
        return []

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        weight = self.param("weight", nn.initializers.ones, (self.ndim,))
        bias = (
            self.param("bias", nn.initializers.zeros, (self.ndim,))
            if self.bias
            else None
        )

        # Cast to float32 for numerical stability
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.var(x_f32, axis=-1, keepdims=True)
        x_norm = (x_f32 - mean) / jnp.sqrt(var + self.eps)

        # Cast back to compute dtype
        x_norm = x_norm.astype(self.dtype)
        weight_cast = weight.astype(self.dtype)

        if bias is not None:
            bias_cast = bias.astype(self.dtype)
            return weight_cast * x_norm + bias_cast
        else:
            return weight_cast * x_norm
