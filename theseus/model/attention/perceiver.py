"""
Perceiver Resampler: compresses variable-length input into K fixed-size vectors.

Following Flamingo (Alayrac et al., 2022) and Perceiver (Jaegle et al., 2021):
- Learned latent query vectors cross-attend to [queries; raw_input]
- L_res layers of cross-attention + FFN
- Produces fixed-size (K, d) output regardless of input length
"""

from typing import Optional, Tuple, Any, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.module import Module


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver resampler layer: cross-attention + FFN."""

    n_embd: int
    n_head: int
    dropout: float = 0.0
    param_dtype: Any = jnp.float32
    activation_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        z: jax.Array,
        kv_input: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        """
        Args:
            z: (B, K, C) latent queries
            kv_input: (B, K+L, C) concatenation of [z; raw_input]
            deterministic: whether to apply dropout

        Returns:
            (B, K, C) updated latent queries
        """
        head_dim = self.n_embd // self.n_head

        # Pre-norm
        z_normed = nn.LayerNorm(param_dtype=self.param_dtype)(z)
        kv_normed = nn.LayerNorm(param_dtype=self.param_dtype)(kv_input)

        # Cross-attention: Q from z, K/V from [z; raw_input]
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
        )(kv_normed)
        v = nn.Dense(
            self.n_embd,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.activation_dtype,
        )(kv_normed)

        B = z.shape[0]
        K_q = z.shape[1]
        K_kv = kv_input.shape[1]

        q = q.reshape(B, K_q, self.n_head, head_dim).astype(self.activation_dtype)
        k = k.reshape(B, K_kv, self.n_head, head_dim).astype(self.activation_dtype)
        v = v.reshape(B, K_kv, self.n_head, head_dim).astype(self.activation_dtype)

        # Attention (no causal mask — full bidirectional)
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

        # FFN with pre-norm
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


class PerceiverResampler(Module):
    """Perceiver resampler that compresses variable-length input to K fixed vectors.

    Following Flamingo: learned queries cross-attend to [queries; raw_input].
    This produces a fixed-size output regardless of input length.
    """

    n_embd: int = field("architecture/n_embd", default=2048)
    n_latents: int = field("architecture/sidechannel/n_latents", default=128)
    perceiver_layers: int = field(
        "architecture/sidechannel/perceiver_layers", default=2
    )
    perceiver_heads: int = field("architecture/sidechannel/perceiver_heads", default=8)
    dropout: float = field("architecture/dropout", default=0.0)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        # Learned latent query vectors
        self.latent_queries = self.param(
            "latent_queries",
            nn.initializers.normal(stddev=0.02),
            (self.n_latents, self.n_embd),
            self._param_dtype,
        )

        # Resampler layers
        self.layers = [
            PerceiverResamplerLayer(
                n_embd=self.n_embd,
                n_head=self.perceiver_heads,
                dropout=self.dropout,
                param_dtype=self._param_dtype,
                activation_dtype=self._activation_dtype,
            )
            for _ in range(self.perceiver_layers)
        ]

        # Final layer norm
        self.ln_out = nn.LayerNorm(param_dtype=self._param_dtype)

    def __call__(
        self,
        raw_input: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        """Compress variable-length input to fixed K vectors.

        Args:
            raw_input: (B, L, C) variable-length encoded input embeddings

        Returns:
            (B, K, C) compressed channel state
        """
        B = raw_input.shape[0]

        # Broadcast learned queries to batch dimension
        z = jnp.broadcast_to(
            self.latent_queries[None, :, :].astype(self._activation_dtype),
            (B, self.n_latents, self.n_embd),
        )

        raw_input = raw_input.astype(self._activation_dtype)

        for layer in self.layers:
            # Concatenate [z; raw_input] for KV (Flamingo-style)
            kv_input = jnp.concatenate([z, raw_input], axis=1)
            z = layer(z, kv_input, deterministic=deterministic)

        z = self.ln_out(z)
        return z  # type: ignore[no-any-return]
