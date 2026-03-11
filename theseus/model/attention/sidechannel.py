"""
Grouped Sidechannel Cross-Attention with tanh-gating and KV caching.

Implements the gated cross-attention mechanism from DMA-CoT:
- Q from decoder hidden state, K/V from Perceiver-compressed channel states
- NO RoPE (channel states are semantic summaries, not sequential tokens)
- Tanh-gated output: output = tanh(alpha) * xattn_result (alpha init=0)
- Per-position channel selection via sidechannel_mask for blockwise masking
- Separate KV cache in "cross_cache" collection for inference
"""

import math
from typing import Optional, Tuple, Any, List, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module


class GroupedSidechannelCrossAttention(Module):
    """Gated cross-attention for side-channel inputs.

    Based on GroupedSelfAttention projection patterns but specialized for
    cross-attention: Q from decoder, K/V from external channel states.

    The tanh gate is initialized to 0, so at initialization this module
    contributes nothing to the residual stream, preserving pretrained
    model behavior exactly.
    """

    n_embd: int = field("architecture/n_embd", default=2048)
    n_layers: int = field("architecture/n_layers", default=32)
    n_head: int = field("architecture/sidechannel/n_head", default=-1)
    n_kv_head: int = field("architecture/sidechannel/n_kv_head", default=-1)
    n_latents: int = field("architecture/sidechannel/n_latents", default=128)
    dropout: float = field("architecture/dropout", default=0.0)
    bias: bool = field("architecture/bias", default=True)
    attn_bias: bool = field("architecture/sidechannel/attn_bias", default=False)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        # Default n_head to architecture n_head if not specified
        n_head_main: int = field("architecture/n_head", default=16)
        n_head = self.n_head if self.n_head > 0 else n_head_main
        n_kv_head = self.n_kv_head if self.n_kv_head > 0 else n_head

        assert self.n_embd % n_head == 0
        head_dim = self.n_embd // n_head
        assert n_head % n_kv_head == 0
        n_rep = n_head // n_kv_head

        self.head_dim = head_dim
        self.n_head_eff = n_head
        self.n_kv_head_eff = n_kv_head
        self.n_rep = n_rep

        kernel_init_std = 0.02
        proj_init_std = 0.02 / math.sqrt(2 * self.n_layers)

        # Q projection (from decoder hidden states)
        self.q_proj = nn.Dense(
            n_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        # K projection (from channel states)
        self.k_proj = nn.Dense(
            n_kv_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        # V projection (from channel states)
        self.v_proj = nn.Dense(
            n_kv_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        # Output projection
        self.o_proj = nn.Dense(
            self.n_embd,
            use_bias=self.attn_bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=proj_init_std),
                (Axes.N_ATTN.value, Axes.N_EMBD.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        # Tanh gate initialized to 0 => tanh(0) = 0 => no contribution at init
        self.gate = self.param(
            "gate",
            nn.initializers.zeros,
            (1,),
            self._param_dtype,
        )

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Repeat KV heads to match query heads for GQA."""
        if self.n_rep == 1:
            return x
        b, t, kvh, d = x.shape
        x = x[:, :, :, None, :]
        x = jnp.broadcast_to(x, (b, t, kvh, self.n_rep, d))
        return x.reshape(b, t, kvh * self.n_rep, d)

    @nn.compact
    def _cached_cross_kv(
        self,
        k: jax.Array,
        v: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Cache cross-attention K/V in 'cross_cache' collection.

        Channel states change infrequently, so we cache their projections.
        K, V shape: (B, K_latents, n_kv_head, D)
        """
        if not self.is_mutable_collection("cross_cache"):
            return k, v

        cached_key = self.variable(
            "cross_cache", "cached_key", jnp.zeros, k.shape, k.dtype
        )
        cached_value = self.variable(
            "cross_cache", "cached_value", jnp.zeros, v.shape, v.dtype
        )

        cached_key.value = k
        cached_value.value = v

        return k, v

    def __call__(
        self,
        x: jax.Array,
        channel_states: jax.Array,
        channel_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        """Gated cross-attention to side-channel states.

        Args:
            x: (B, T, C) decoder hidden states
            channel_states: (B, N, K, C) N channels with K latent vectors each
            channel_mask: (B, T) int in [0..N-1], which channel each position
                         attends to. Enables blockwise masking during training.
                         If None, all positions attend to channel 0.
            deterministic: whether to apply dropout

        Returns:
            (B, T, C) gated cross-attention output (to be added to residual)
        """
        B, T, C = x.shape
        K = channel_states.shape[2]

        # Select per-position channel state using channel_mask
        if channel_mask is not None:
            # channel_mask: (B, T) int -> index into channel_states
            batch_idx = jnp.arange(B)[:, None]  # (B, 1)
            # selected: (B, T, K, C) — each position gets its assigned channel
            selected_channels = channel_states[batch_idx, channel_mask]  # (B, T, K, C)
        else:
            # Default: attend to channel 0
            selected_channels = jnp.broadcast_to(
                channel_states[:, 0:1, :, :], (B, T, K, C)
            )

        # Project Q from decoder hidden states
        q = self.q_proj(x).reshape(B, T, self.n_head_eff, self.head_dim)

        # Project K, V from selected channel states
        # Reshape: (B, T, K, C) -> (B*T, K, C) for projection, then back
        selected_flat = selected_channels.reshape(B * T, K, C)
        k = self.k_proj(selected_flat).reshape(
            B, T, K, self.n_kv_head_eff, self.head_dim
        )
        v = self.v_proj(selected_flat).reshape(
            B, T, K, self.n_kv_head_eff, self.head_dim
        )

        # Repeat KV heads for GQA
        # k, v: (B, T, K, n_kv_head, D) -> need (B, T, K, n_head, D)
        if self.n_rep > 1:
            k = jnp.broadcast_to(
                k[:, :, :, :, None, :],
                (B, T, K, self.n_kv_head_eff, self.n_rep, self.head_dim),
            ).reshape(B, T, K, self.n_head_eff, self.head_dim)
            v = jnp.broadcast_to(
                v[:, :, :, :, None, :],
                (B, T, K, self.n_kv_head_eff, self.n_rep, self.head_dim),
            ).reshape(B, T, K, self.n_head_eff, self.head_dim)

        # Attention: each position's Q (1 vector) attends to K latent vectors K, V
        # q: (B, T, H, D), k: (B, T, K, H, D), v: (B, T, K, H, D)
        # Reshape for per-position attention:
        # q: (B*T, 1, H, D), k: (B*T, K, H, D), v: (B*T, K, H, D)
        q_flat = q.reshape(B * T, 1, self.n_head_eff, self.head_dim)
        k_flat = k.reshape(B * T, K, self.n_head_eff, self.head_dim)
        v_flat = v.reshape(B * T, K, self.n_head_eff, self.head_dim)

        q_flat = q_flat.astype(self._activation_dtype)
        k_flat = k_flat.astype(self._activation_dtype)
        v_flat = v_flat.astype(self._activation_dtype)

        # dot_product_attention: (B*T, 1, H, D) x (B*T, K, H, D) -> (B*T, 1, H, D)
        y = jax.nn.dot_product_attention(q_flat, k_flat, v_flat)

        # Reshape back: (B*T, 1, H, D) -> (B, T, H, D) -> (B, T, C)
        y = y.reshape(B, T, self.n_head_eff, self.head_dim)
        y = y.reshape(B, T, C)

        # Output projection
        y = self.o_proj(y)

        # Apply dropout
        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        # Apply tanh gate
        gate_val = jnp.tanh(self.gate)
        y = gate_val * y

        return y
