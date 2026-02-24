"""
Base self-attention module with hook-based API and KV cache support.

All Q/K/V hooks operate on (B, T, H, D) format.
Subclasses override _project_inner/preprocess_qkv/build_mask/attn/postprocess_attn/output_proj.

KV cache is activated by calling model.apply(..., mutable=['cache']).
Without mutable=['cache'], the cache is a no-op (training mode).
"""

import math
from typing import Tuple, Any, Optional, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.model.masks import cache_mask


class SelfAttention(Module):
    n_embd: int = field("architecture/n_embd", default=2048)
    n_layers: int = field("architecture/n_layers", default=32)
    bias: bool = field("architecture/bias", default=True)
    dropout: float = field("architecture/dropout", default=0.0)

    n_head: int = field("architecture/n_head", default=16)
    block_size: int = field("architecture/block_size", default=512)

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
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.n_layers)),
                (Axes.N_EMBD_OUT.value, Axes.N_EMBD.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

    # ------------------------------------------------------------------
    # Projection hooks
    # ------------------------------------------------------------------

    def _project_inner(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Raw Q/K/V projection. Override in subclasses. Returns (B, T, H, D)."""
        B, T, _C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.reshape(B, T, self.n_head, self.head_dim)
        k = k.reshape(B, T, self.n_head, self.head_dim)
        v = v.reshape(B, T, self.n_head, self.head_dim)
        return q, k, v

    def project(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Project input to (q, k, v). Calls _project_inner."""
        return self._project_inner(x)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    @nn.compact
    def _cached_kv(
        self, k: jax.Array, v: jax.Array
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Update KV cache if active. k, v: (B, T, H, D).

        Returns (k, v, cache_index_after_update) where cache_index is None
        when cache is not active (training mode).
        """
        if not self.is_mutable_collection("cache"):
            return k, v, None

        # Cache is requested — create or update variables
        # Allocate to block_size so we can decode up to that many tokens
        B, _T, H, D = k.shape
        cache_shape = (B, self.block_size, H, D)
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, cache_shape, k.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, cache_shape, v.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if self.has_variable("cache", "cache_index"):
            # Decode step: k, v are (B, 1, H, D) — single new token
            cur_index = cache_index.value
            batch_dims = k.ndim - 3  # typically 1 (B)
            zero = jnp.array(0, dtype=jnp.int32)
            indices: tuple[jax.Array, ...] = (zero,) * batch_dims + (
                cur_index,
                zero,
                zero,
            )
            k = jax.lax.dynamic_update_slice(cached_key.value, k, indices)
            v = jax.lax.dynamic_update_slice(cached_value.value, v, indices)
            cached_key.value = k
            cached_value.value = v
            new_index = cur_index + 1
            cache_index.value = new_index
            return k, v, new_index
        else:
            # Prefill: k, v are (B, T_prefill, H, D) — write into the larger cache
            T_prefill = k.shape[1]
            batch_dims = k.ndim - 3
            zero = jnp.array(0, dtype=jnp.int32)
            indices = (zero,) * batch_dims + (zero, zero, zero)
            new_cached_k = jax.lax.dynamic_update_slice(cached_key.value, k, indices)
            new_cached_v = jax.lax.dynamic_update_slice(cached_value.value, v, indices)
            cached_key.value = new_cached_k
            cached_value.value = new_cached_v
            cache_index.value = jnp.array(T_prefill, dtype=jnp.int32)
            # Return original k, v (not full cache) — prefill uses normal causal mask
            return k, v, None

    # ------------------------------------------------------------------
    # Attention hooks
    # ------------------------------------------------------------------

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
        ci = kwargs.get("_cache_index")
        if ci is not None:
            return cache_mask(t, ci)
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
        """Core attention. Input q/k/v: (B, T_q/T_kv, H, D). Output: (B, T_q, H, D)."""
        # dot_product_attention expects (B, T, H, D) — same as our convention
        # Compute attention in float32 for precision parity between
        # full-sequence and cached single-token paths
        q = q.astype(self._activation_dtype)
        k = k.astype(self._activation_dtype)
        v = v.astype(self._activation_dtype)

        if mask is not None:
            # Use additive bias (-inf for masked) instead of boolean mask
            # to ensure identical DPA kernel path as is_causal=True
            bias = jnp.where(mask, 0.0, -1e9)
            y = jax.nn.dot_product_attention(q, k, v, bias=bias)
        else:
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        return y

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
        B, T, C = x.shape

        q, k, v = self.project(x)

        # For decode steps with cache, inject correct RoPE positions
        if self.has_variable("cache", "cache_index"):
            ci: Any = self.get_variable("cache", "cache_index")
            kwargs = {**kwargs, "positions": jnp.arange(T) + ci}

        q, k, v = self.preprocess_qkv(q, k, v, **kwargs)
        k, v, cache_idx = self._cached_kv(k, v)

        T_kv = k.shape[1]
        mask = self.build_mask(T_kv, padding_mask, _cache_index=cache_idx, **kwargs)
        y = self.attn(q, k, v, mask, **kwargs)
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)

        y = y.reshape(B, T, C)
        y = self.output_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y
