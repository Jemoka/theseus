"""
Attention module with forking support for Thoughtbubbles.
Extends RopeAttention with cumulative scores and token index tracking.
"""

from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.attention.rope import RopeAttention

ATTN_DTYPE = jnp.bfloat16


@jax.jit
def causal_bias(max_block_size: int) -> jax.Array:
    neg = jnp.array(-jnp.inf, ATTN_DTYPE)
    m = jnp.tril(jnp.ones((max_block_size, max_block_size), dtype=jnp.bool_))
    return jnp.where(m, jnp.array(0, ATTN_DTYPE), neg)[None, None, :, :]


@jax.jit
def key_padding_bias(padding_mask: jax.Array) -> jax.Array:
    neg = jnp.array(-jnp.inf, ATTN_DTYPE)
    keep = padding_mask.astype(jnp.bool_)[:, None, None, :]
    return jnp.where(keep, jnp.array(0, ATTN_DTYPE), neg)


class ForkingAttention(RopeAttention):
    """Self-attention with forking support, extending RopeAttention."""

    block_size: int = field("architecture/block_size", default=512)
    max_block_size: int = field("architecture/max_block_size", default=1024)
    use_fork_channel: bool = field("architecture/use_fork_channel", default=True)

    def preprocess_qkv(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Apply RoPE with fractional rotations and fork channel injection."""
        cumulative_scores: Optional[jax.Array] = kwargs.get("cumulative_scores")
        token_index: Optional[jax.Array] = kwargs.get("token_index")

        if token_index is not None:
            # Compute fractional rotations based on token counts
            token_counts = jnp.zeros(
                (*token_index.shape[:-1], self.block_size), dtype=token_index.dtype
            )
            token_counts = token_counts.at[
                jnp.arange(token_counts.shape[0])[:, None], token_index
            ].add(1)

            partial_rotations = jnp.cumsum(
                jnp.take_along_axis(1 / (token_counts + 1e-10), token_index, axis=-1),
                axis=-1,
            )
            q, k = self.rope(q, k, t=partial_rotations)
        else:
            # Standard RoPE
            q, k = self.rope(q, k)

        # Fork-specific: use last channel for cumulative scores
        if self.use_fork_channel and cumulative_scores is not None:
            q = q.at[:, :, :, -1].set(jnp.ones_like(q[:, :, :, -1]))
            k = k.at[:, :, :, -1].set(
                jnp.repeat(cumulative_scores[:, None, :], k.shape[1], axis=1)
            )

        # Attenuate v with cumulative scores
        if cumulative_scores is not None:
            v = jnp.einsum("bnlh,bl->bnlh", v, jnp.exp(cumulative_scores))

        return q, k, v

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute attention with custom masking for forked tokens."""
        token_index: Optional[jax.Array] = kwargs.get("token_index")
        padding_mask: Optional[jax.Array] = kwargs.get("padding_mask")

        q = q.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        k = k.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        v = v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)

        if token_index is not None and padding_mask is not None:
            # Build custom mask for forked tokens
            padding_bias = key_padding_bias(padding_mask)
            padding_bias = jnp.take_along_axis(
                padding_bias, token_index[:, None, None, :], axis=-1
            )
            bias = (
                causal_bias(self.max_block_size)[:, :, : q.shape[1], : k.shape[1]]
                + padding_bias
            )
            y = jax.nn.dot_product_attention(q, k, v, mask=(bias == 0))
        elif token_index is not None:
            bias = causal_bias(self.max_block_size)[:, :, : q.shape[1], : k.shape[1]]
            y = jax.nn.dot_product_attention(q, k, v, bias=bias)
        elif mask is not None:
            y = jax.nn.dot_product_attention(q, k, v, mask=mask, is_causal=True)
        else:
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        return y

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """
        Forward pass with forking support.

        Args:
            x: Input tensor of shape (B, T, C).
            padding_mask: Boolean tensor of shape (B, T). True for valid tokens.
            deterministic: If False, applies dropout.
            **kwargs: Must include cumulative_scores and token_index for forking.

        Returns:
            Output tensor of shape (B, T, C).
        """
        cumulative_scores: Optional[jax.Array] = kwargs.get("cumulative_scores")
        token_index: Optional[jax.Array] = kwargs.get("token_index")

        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = self.preprocess_qkv(
            q,
            k,
            v,
            cumulative_scores=cumulative_scores,
            token_index=token_index,
        )

        y = self.attn(
            q,
            k,
            v,
            mask=padding_mask[:, None, None, :] if padding_mask is not None else None,
            token_index=token_index,
            padding_mask=padding_mask,
        )

        if padding_mask is not None and token_index is not None:
            padding_mask_indexed = jnp.take_along_axis(
                padding_mask, token_index, axis=-1
            )
            y = y * padding_mask_indexed[:, :, None, None].astype(y.dtype)
        elif padding_mask is not None:
            y = y * padding_mask[:, :, None, None].astype(y.dtype)

        if not deterministic:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        y = y.reshape(B, T, C)
        y = self.c_proj(y)

        if not deterministic:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y
