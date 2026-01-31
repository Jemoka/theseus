import jax
import flax.linen as nn
import jax.numpy as jnp

from theseus.config import field


class LayerNorm(nn.Module):
    ndim: int = field("architecture/n_embd")
    bias: bool = field("architecture/bias")
    dtype: jnp.dtype = jnp.bfloat16

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
        x_norm = (x_f32 - mean) / jnp.sqrt(var + 1e-5)

        # Cast back to compute dtype
        x_norm = x_norm.astype(self.dtype)
        weight_cast = weight.astype(self.dtype)

        if bias is not None:
            bias_cast = bias.astype(self.dtype)
            return weight_cast * x_norm + bias_cast
        else:
            return weight_cast * x_norm
