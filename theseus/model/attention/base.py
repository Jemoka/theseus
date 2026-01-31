"""
Your grandpa's vanilla self-attention module.
"""

import math
from typing import Tuple, Dict, Any, Optional, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module

ATTN_DTYPE = jnp.bfloat16


class SelfAttention(Module):
    n_embd: int = field("architecture/n_embd")
    n_head: int = field("architecture/n_head")
    n_layers: int = field("architecture/n_layers")
    bias: bool = field("architecture/bias")
    dropout: float = field("architecture/dropout")

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    def setup(self) -> None:
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head

        self.c_attn = nn.Dense(
            3 * self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )

        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.n_layers)),
                (Axes.N_EMBD_OUT.value, Axes.N_EMBD.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Dict[Any, Any]
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return q, k, v

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Dict[Any, Any],
    ) -> jax.Array:
        q = q.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        k = k.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        v = v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)

        if mask is not None:
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
        **kwargs: Dict[Any, Any],
    ) -> jax.Array:
        """
        Args:
            x: Input tensor of shape (B, T, C).
            padding_mask: Boolean tensor of shape (B, T). True for valid tokens,
                False for padding tokens. When provided, padded positions are
                masked in attention and zeroed in output.
            deterministic: If False, applies dropout.
            **kwargs: Additional arguments passed to preprocess_qkv and attn.

        Returns:
            Output tensor of shape (B, T, C).
        """

        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = self.preprocess_qkv(q, k, v, **kwargs)

        if padding_mask is not None:
            mask = padding_mask[:, None, None, :]  # (B, 1, 1, T)
        else:
            mask = None

        y = self.attn(q, k, v, mask, **kwargs)

        if padding_mask is not None:
            y = y * padding_mask[:, :, None, None].astype(y.dtype)

        if not deterministic:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        y = y.reshape(B, T, C)
        y = self.c_proj(y)

        if not deterministic:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y
