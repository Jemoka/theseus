import math
import jax
import flax.linen as nn
import jax.numpy as jnp

from theseus.config import field


class MLP(nn.Module):
    n_embd: int = field("architecture/n_embd")
    n_layers: int = field("architecture/n_layers")
    bias: bool = field("architecture/bias")
    dropout: float = field("architecture/dropout")

    def setup(self) -> None:
        self.c_fc = nn.Dense(
            4 * self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02), ("n_embd", "n_embd_ff")
            ),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )

        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.n_layers)),
                ("n_embd_ff", "n_embd"),
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
