"""
Base self-attention module with hook-based API.

All Q/K/V hooks operate on (B, T, H, D) format.
Subclasses override project/preprocess_qkv/build_mask/attn/postprocess_attn/output_proj.
"""

import math
from typing import Tuple, Any, Optional, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module

ATTN_DTYPE = jnp.bfloat16


class SelfAttention(Module):
    n_embd: int = field("architecture/n_embd")
    n_layers: int = field("architecture/n_layers")
    bias: bool = field("architecture/bias")
    dropout: float = field("architecture/dropout")

    n_head: int = field("architecture/n_head", default=16)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
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

    def project(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Project input to (q, k, v) each in (B, T, H, D) format."""
        B, T, _C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.reshape(B, T, self.n_head, self.head_dim)
        k = k.reshape(B, T, self.n_head, self.head_dim)
        v = v.reshape(B, T, self.n_head, self.head_dim)
        return q, k, v

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Hook for RoPE, KV repeat, etc. Input/output: (B, T, H, D)."""
        return q, k, v

    def build_mask(
        self,
        t: int,
        padding_mask: Optional[jax.Array],
        **kwargs: Any,
    ) -> Optional[jax.Array]:
        """Construct attention mask. Returns bool mask or None."""
        if padding_mask is not None:
            return padding_mask[:, None, None, :]  # (B, 1, 1, T)
        return None

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Core attention. Input q/k/v: (B, T, H, D). Output: (B, T, H, D)."""
        # Transpose to (B, H, T, D) for dot_product_attention
        q = q.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        k = k.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)
        v = v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)

        if mask is not None:
            y = jax.nn.dot_product_attention(q, k, v, mask=mask, is_causal=True)
        else:
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        # Transpose back to (B, T, H, D)
        return y.transpose(0, 2, 1, 3)

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        """Post-attention processing. Input/output: (B, T, H, D)."""
        if padding_mask is not None:
            y = y * padding_mask[:, :, None, None].astype(y.dtype)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y

    def output_proj(self, y: jax.Array) -> jax.Array:
        """Output projection. (B, T, C) → (B, T, C)."""
        return self.c_proj(y)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """
        Args:
            x: Input tensor of shape (B, T, C).
            padding_mask: Boolean tensor of shape (B, T) or (B, 1, T, T).
                True for valid tokens, False for padding.
            deterministic: If False, applies dropout.
            **kwargs: Additional arguments passed to hooks.

        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape

        q, k, v = self.project(x)
        q, k, v = self.preprocess_qkv(q, k, v, **kwargs)
        mask = self.build_mask(T, padding_mask, **kwargs)
        y = self.attn(q, k, v, mask, **kwargs)
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)

        y = y.reshape(B, T, C)
        y = self.output_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y
