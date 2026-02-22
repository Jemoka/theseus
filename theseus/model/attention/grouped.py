"""
Grouped Query Attention (GQA) with RoPE, sliding window, and partial rotation support.
Inherits from SelfAttention, overriding projection, RoPE, masking, and attention.
"""

import math
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.attention.base import SelfAttention
from theseus.model.layers.rope import RotaryPosEncoding
from theseus.model.masks import (
    causal_mask,
    sliding_window_mask,
    combine_padding,
    cache_mask,
)

ATTN_DTYPE = jnp.bfloat16


class GroupedSelfAttention(SelfAttention):
    n_embd: int = field("architecture/n_embd", default=4096)
    n_layers: int = field("architecture/n_layers", default=32)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=-1)  # -1 -> use n_head
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    rope_theta: float = field("architecture/rope_theta", default=1e6)
    partial_rotary_factor: float = field(
        "architecture/partial_rotary_factor", default=1.0
    )
    use_sliding_window: bool = field("architecture/use_sliding_window", default=False)
    sliding_window: int = field(
        "architecture/sliding_window", default=-1
    )  # -1 -> no sliding
    bias: bool = field("architecture/bias", default=True)
    attn_bias: bool = field("architecture/attn_bias", default=True)

    def setup(self) -> None:
        assert self.n_embd % self.n_head == 0
        head_dim = self.n_embd // self.n_head
        n_kv_head = self.n_head if self.n_kv_head == -1 else self.n_kv_head
        assert self.n_head % n_kv_head == 0
        n_rep = self.n_head // n_kv_head
        self.head_dim = head_dim
        self.n_kv_head_eff = n_kv_head
        self.n_rep = n_rep

        kernel_init_std = 0.02
        proj_init_std = 0.02 / math.sqrt(2 * self.n_layers)

        self.q_proj = nn.Dense(
            self.n_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=jnp.float32,
            dtype=ATTN_DTYPE,
        )
        self.k_proj = nn.Dense(
            n_kv_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=jnp.float32,
            dtype=ATTN_DTYPE,
        )
        self.v_proj = nn.Dense(
            n_kv_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=jnp.float32,
            dtype=ATTN_DTYPE,
        )
        self.o_proj = nn.Dense(
            self.n_embd,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=proj_init_std),
                (Axes.N_ATTN.value, Axes.N_EMBD.value),
            ),
            param_dtype=jnp.float32,
            dtype=ATTN_DTYPE,
        )

        self.rope = RotaryPosEncoding(
            head_dim,
            base=int(self.rope_theta),
            seq_dim=1,
            partial_rotary_factor=self.partial_rotary_factor,
        )

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.n_rep == 1:
            return x
        b, t, kvh, d = x.shape
        x = x[:, :, :, None, :]
        x = jnp.broadcast_to(x, (b, t, kvh, self.n_rep, d))
        return x.reshape(b, t, kvh * self.n_rep, d)

    def _project_inner(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.n_kv_head_eff, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_kv_head_eff, self.head_dim)
        return q, k, v

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q, k = self.rope(q, k, t=kwargs.get("positions"))
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        return q, k, v

    def build_mask(
        self,
        t: int,
        padding_mask: Optional[jax.Array],
        **kwargs: Any,
    ) -> Optional[jax.Array]:
        # When cache is active, _cache_index is passed from __call__
        ci = kwargs.get("_cache_index")
        if ci is not None:
            return cache_mask(t, ci)
        # Accept an already-prepared 4D mask (True=keep) for parity/debug
        if padding_mask is not None and padding_mask.ndim == 4:
            return padding_mask
        sliding = kwargs.get("sliding", False)
        base = (
            sliding_window_mask(t, self.sliding_window)
            if sliding and self.sliding_window > 0
            else causal_mask(t)
        )
        return combine_padding(base, padding_mask)

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        b = q.shape[0]
        t_q = q.shape[1]
        t_kv = k.shape[1]
        qh = q.transpose(0, 2, 1, 3).astype(jnp.float32)
        kh = k.transpose(0, 2, 1, 3).astype(jnp.float32)
        vh = v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)

        scores = jnp.einsum("bhtd,bhTd->bhtT", qh, kh)
        scores = scores / jnp.sqrt(self.head_dim).astype(jnp.float32)

        if mask is not None:
            bias = jnp.where(
                jnp.broadcast_to(mask, (b, 1, t_q, t_kv)), 0.0, -1e9
            ).astype(jnp.float32)
            scores = scores + bias

        attn_w = jax.nn.softmax(scores, axis=-1).astype(vh.dtype)
        y = jnp.einsum("bhtT,bhTd->bhtd", attn_w, vh.astype(attn_w.dtype))
        return y.transpose(0, 2, 1, 3)  # back to (B, T_q, H, D)

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        # No-op: dropout is handled in the shared __call__ after output_proj
        return y

    def output_proj(self, y: jax.Array) -> jax.Array:
        return self.o_proj(y)
