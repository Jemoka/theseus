import math
from typing import Optional, Tuple, Any, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.model.layers.rope import RotaryPosEncoding
from theseus.model.masks import causal_mask, sliding_window_mask, combine_padding

ATTN_DTYPE = jnp.float32


class GroupedSelfAttention(Module):
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
    attn_bias: bool = field("architecture/attn_bias", default=True)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

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

    def _select_mask(
        self, t: int, padding_mask: Optional[jnp.ndarray], sliding: bool
    ) -> Optional[jnp.ndarray]:
        base = (
            sliding_window_mask(t, self.sliding_window)
            if sliding and self.sliding_window > 0
            else causal_mask(t)
        )
        base = combine_padding(base, padding_mask)
        return base  # boolean

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        sliding: bool = False,
        positions: Optional[jnp.ndarray] = None,
    ) -> jax.Array:
        b, t, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(b, t, self.n_head, self.head_dim)
        k = k.reshape(b, t, self.n_kv_head_eff, self.head_dim)
        v = v.reshape(b, t, self.n_kv_head_eff, self.head_dim)

        # apply RoPE before kv repeat (B, T, H, D)
        q, k = self.rope(q, k, t=positions)
        qh = q.transpose(0, 2, 1, 3)  # (B, H, T, D)
        kh = k.transpose(0, 2, 1, 3)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        kh = k.transpose(0, 2, 1, 3)
        vh = v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE)

        # Accept an already-prepared 4D mask (True=keep) to bypass internal construction (used for parity/debug).
        mask: Optional[jnp.ndarray]
        if padding_mask is not None and padding_mask.ndim == 4:
            mask = padding_mask
        else:
            mask = self._select_mask(t, padding_mask, sliding)  # (1,1,T,T) or None
        bias = None
        if mask is not None:
            # convert to additive bias float32 pre-softmax
            mask = jnp.broadcast_to(mask, (b, 1, t, t))
            bias = jnp.where(mask, 0.0, -1e9).astype(jnp.float32)

        qh_f32 = qh.astype(jnp.float32)
        kh_f32 = kh.astype(jnp.float32)
        scores = jnp.einsum("bhtd,bhTd->bhtT", qh_f32, kh_f32)
        scores = scores / jnp.sqrt(self.head_dim).astype(jnp.float32)
        if bias is not None:
            scores = scores + bias
        attn_w = jax.nn.softmax(scores, axis=-1).astype(vh.dtype)

        attn_out = jnp.einsum("bhtT,bhTd->bhtd", attn_w, vh.astype(attn_w.dtype))

        attn = attn_out.transpose(0, 2, 1, 3).reshape(b, t, self.n_embd)

        attn = self.o_proj(attn)

        if not deterministic and self.dropout > 0:
            attn = nn.Dropout(rate=self.dropout)(attn, deterministic=False)

        return attn
