from typing import Optional, Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.attention.forking import (
    ForkingAttention,
    ATTN_DTYPE,
    key_padding_bias,
)


class ScratchSparseCrossAttention(ForkingAttention):
    scratch_head_dim: int = field("architecture/scratch_head_dim", default=64)

    def setup(self) -> None:
        self.q_proj = nn.Dense(
            self.scratch_head_dim,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_SCRATCH.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.k_proj = nn.Dense(
            self.scratch_head_dim,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_SCRATCH.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        token_index: jax.Array = kwargs.get("query_token_index")  # type: ignore
        if kwargs.get("token_index") is not None and token_index is not None:
            del kwargs["token_index"]
        return super().postprocess_attn(
            y, padding_mask, deterministic, token_index=token_index, **kwargs
        )

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Cross-attention with causal mask based on token positions, not array indices.

        The inherited ForkingAttention.attn uses a standard lower-triangular mask
        on array positions, which is incorrect for cross-attention where Q (post
        top-k selection) and K (original sequence) have different position mappings.
        This override builds the causal mask from actual token indices so that a
        query at original position p can only attend to keys at positions <= p.
        """
        cumulative_scores: jax.Array = kwargs.get("cumulative_scores")  # type: ignore
        token_index: jax.Array = kwargs.get("token_index")  # type: ignore
        query_token_index: jax.Array = kwargs.get("query_token_index")  # type: ignore

        if (
            cumulative_scores is None
            or token_index is None
            or query_token_index is None
        ):
            raise ValueError(
                "cumulative_scores, token_index, and query_token_index must be "
                "provided for ScratchSparseCrossAttention"
            )

        # Build causal mask from token positions:
        # query at original position i can attend to key at original position j iff i >= j
        # query_token_index: (B, Q_len), token_index: (B, K_len)
        causal_mask = (
            query_token_index[:, :, None] >= token_index[:, None, :]
        )  # (B, Q_len, K_len)
        causal_mask = causal_mask[
            :, None, :, :
        ]  # (B, 1, Q_len, K_len) for head broadcasting

        if mask is not None:
            padding_bias = key_padding_bias(mask)  # (B, 1, 1, T_mask)
            padding_bias = jnp.take_along_axis(
                padding_bias, token_index[:, None, None, :], axis=-1
            )
            padding_keep = padding_bias == 0  # True where valid
            causal_mask = causal_mask & padding_keep

        # Scale v by cumulative scores
        v = jnp.einsum("bthd,bt->bthd", v, jnp.exp(cumulative_scores))

        q = q.astype(ATTN_DTYPE)
        k = k.astype(ATTN_DTYPE)
        v = v.astype(ATTN_DTYPE)

        # Manual attention: Q/K may have different last dim than V
        # (Q/K are projected to scratch_head_dim, V stays at n_embd)
        scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=ATTN_DTYPE))
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        attn_weights = jnp.where(
            causal_mask.transpose(0, 1, 2, 3),  # (B, 1, Q, K)
            attn_weights,
            jnp.array(-1e9, dtype=ATTN_DTYPE),
        )
        self.sow("plots", "scratching_attn_weights", attn_weights)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        y = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        return y

    @nn.compact
    def __call__(  # type: ignore[override]
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        cumulative_scores: jax.Array = kwargs.get("cumulative_scores")  # type: ignore

        # Project Q and K to lower dimension; V stays unprojected
        q = self.q_proj(q)  # (B, T_q, scratch_head_dim)
        k = self.k_proj(k)  # (B, T_k, scratch_head_dim)

        # Fork channel weighting (identical to ForkingAttention.preprocess_qkv)
        if self.use_fork_channel:
            q = q.at[:, :, -1].set(jnp.ones_like(q[:, :, -1]))
            k = k.at[:, :, -1].set(cumulative_scores)

        # each of these should be a single "head"
        y = self.attn(
            q[:, :, None, :],
            k[:, :, None, :],
            v[:, :, None, :],
            padding_mask,
            **kwargs,
        )
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)

        return y[:, :, 0, :]  # get rid of the additional head channel
