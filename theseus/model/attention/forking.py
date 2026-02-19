"""
Attention module with forking support for Thoughtbubbles.
Extends RopeAttention with cumulative scores and token index tracking.
"""

from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp

from theseus.config import field
from theseus.model.attention.rope import RopeAttention

ATTN_DTYPE = jnp.bfloat16


@jax.jit(static_argnums=0)
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
        cumulative_scores: jax.Array = kwargs.get("cumulative_scores")  # type: ignore
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore

        if cumulative_scores is None or token_index is None:
            raise ValueError(
                "cumulative_scores and token_index must be provided for ForkingAttention"
            )

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
        if self.use_fork_channel:
            q = q.at[:, :, :, -1].set(jnp.ones_like(q[:, :, :, -1]))
            k = k.at[:, :, :, -1].set(
                jnp.repeat(cumulative_scores[:, None, :], k.shape[1], axis=1)
            )

        return q, k, v

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        cumulative_scores: jax.Array = kwargs.get("cumulative_scores")  # type: ignore
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore

        if cumulative_scores is None or token_index is None:
            raise ValueError(
                "cumulative_scores and token_index must be provided for ForkingAttention"
            )

        # Build attention bias/mask to exactly mirror the legacy implementation.
        # When padding_mask is provided we convert the additive bias to a boolean mask;
        # otherwise we pass an additive causal bias.
        if mask is not None:
            padding_mask = mask[:, 0, 0, :]
            padding_bias = key_padding_bias(padding_mask)
            padding_bias = jnp.take_along_axis(
                padding_bias, token_index[:, None, None, :], axis=-1
            )
            attn_bias = (
                causal_bias(self.max_block_size)[:, :, : q.shape[-2], : k.shape[-2]]
                + padding_bias
            )
            boolean_mask = attn_bias == 0  # broadcasted lower‑triangular keep mask
            attn_bias = None  # use mask path below
        else:
            attn_bias = causal_bias(self.max_block_size)[
                :, :, : q.shape[-2], : k.shape[-2]
            ]
            boolean_mask = None

        v = jnp.einsum("bnlh,bl->bnlh", v, jnp.exp(cumulative_scores))

        q = q.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        k = k.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        v = v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)

        if boolean_mask is not None:
            # Legacy path: boolean mask (no explicit causal flag).
            y = jax.nn.dot_product_attention(q, k, v, mask=boolean_mask)
        else:
            # Legacy path when no padding mask: additive causal bias.
            y = jax.nn.dot_product_attention(q, k, v, bias=attn_bias)

        return y

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore

        if padding_mask is not None:
            padding_mask = jnp.take_along_axis(padding_mask, token_index, axis=-1)

        return super().postprocess_attn(y, padding_mask, deterministic, **kwargs)
