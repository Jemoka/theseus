"""
Lightweight side-channel encoder: learned queries cross-attend to embedded
side tokens, then refine through MLP. No Flamingo-style KV concatenation.

  x_side = embed(idx_side)          # (B, L, C)
  q_side = learned param            # (K, C)
  q_side = cross_attn(Q=q, KV=x) + q   # compress
  q_side = mlp(q) + q                   # refine
  (repeat n_layers times)
  return layer_norm(q_side)         # (B, K, C)
"""

from typing import Optional, Tuple, Any, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.module import Module


class SideChannelEncoderLayer(nn.Module):
    """Single encoder layer: cross-attention (Q=latents, KV=input) + MLP."""

    n_embd: int
    n_head: int
    dropout: float = 0.0
    param_dtype: Any = jnp.float32
    activation_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        z: jax.Array,
        x: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        """
        Args:
            z: (B, K, C) learned queries
            x: (B, L, C) embedded side-channel tokens
            deterministic: whether to apply dropout

        Returns:
            (B, K, C) updated queries
        """
        head_dim = self.n_embd // self.n_head

        # Pre-norm
        z_normed = nn.LayerNorm(param_dtype=self.param_dtype)(z)
        x_normed = nn.LayerNorm(param_dtype=self.param_dtype)(x)

        # Cross-attention: Q from latent queries, K/V from side-channel tokens
        q = nn.Dense(
            self.n_embd,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(z_normed)
        k = nn.Dense(
            self.n_embd,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(x_normed)
        v = nn.Dense(
            self.n_embd,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(x_normed)

        B = z.shape[0]
        K_q = z.shape[1]
        L_kv = x.shape[1]

        q = q.reshape(B, K_q, self.n_head, head_dim).astype(self.activation_dtype)
        k = k.reshape(B, L_kv, self.n_head, head_dim).astype(self.activation_dtype)
        v = v.reshape(B, L_kv, self.n_head, head_dim).astype(self.activation_dtype)

        # Full bidirectional attention
        attn_out = jax.nn.dot_product_attention(q, k, v)
        attn_out = attn_out.reshape(B, K_q, self.n_embd)

        # Output projection
        attn_out = nn.Dense(
            self.n_embd,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(attn_out)

        if not deterministic and self.dropout > 0:
            attn_out = nn.Dropout(rate=self.dropout)(attn_out, deterministic=False)

        # Residual
        z = z + attn_out

        # MLP with pre-norm
        z_ffn = nn.LayerNorm(param_dtype=self.param_dtype)(z)
        ff_out = nn.Dense(
            4 * self.n_embd,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(z_ffn)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dense(
            self.n_embd,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(ff_out)

        if not deterministic and self.dropout > 0:
            ff_out = nn.Dropout(rate=self.dropout)(ff_out, deterministic=False)

        z = z + ff_out

        return z


class SideChannelEncoder(Module):
    """Lightweight encoder: learned queries cross-attend to side-channel tokens.

    Produces fixed-size (K, C) output regardless of input length L.
    """

    n_embd: int = field("architecture/n_embd", default=2048)
    n_latents: int = field("architecture/sidechannel/n_latents", default=128)
    encoder_layers: int = field(
        "architecture/sidechannel/encoder_layers", default=1
    )
    encoder_heads: int = field("architecture/sidechannel/encoder_heads", default=8)
    dropout: float = field("architecture/dropout", default=0.0)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        self.latent_queries = self.param(
            "latent_queries",
            nn.initializers.normal(stddev=0.02),
            (self.n_latents, self.n_embd),
            self._param_dtype,
        )

        self.layers = [
            SideChannelEncoderLayer(
                n_embd=self.n_embd,
                n_head=self.encoder_heads,
                dropout=self.dropout,
                param_dtype=self._param_dtype,
                activation_dtype=self._activation_dtype,
            )
            for _ in range(self.encoder_layers)
        ]

        self.ln_out = nn.LayerNorm(param_dtype=self._param_dtype)

    def __call__(
        self,
        raw_input: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        """Compress variable-length input to fixed K vectors.

        Args:
            raw_input: (B, L, C) embedded side-channel tokens

        Returns:
            (B, K, C) compressed channel state
        """
        B = raw_input.shape[0]

        z = jnp.broadcast_to(
            self.latent_queries[None, :, :].astype(self._activation_dtype),
            (B, self.n_latents, self.n_embd),
        )

        raw_input = raw_input.astype(self._activation_dtype)

        for layer in self.layers:
            z = layer(z, raw_input, deterministic=deterministic)

        z = self.ln_out(z)
        return z  # type: ignore[no-any-return]


# Backwards compatibility alias
PerceiverResampler = SideChannelEncoder
