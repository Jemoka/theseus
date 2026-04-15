"""Mamba-2 selective state space block.

Implements the Mamba-2 architecture (Dao & Gu, 2024) using the
Structured State Space Duality (SSD) formulation.  The selective
scan is computed via ``jax.lax.associative_scan`` for efficient
parallel execution on accelerators.
"""

import math
from typing import Any, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import configure, field
from theseus.model.axes import Axes
from theseus.model.layers.rmsnorm import RMSNorm
from theseus.model.module import Module


def _selective_scan(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    dt: jax.Array,
    x: jax.Array,
) -> jax.Array:
    """Parallel selective scan via associative scan.

    Args:
        A: (b, T, H)         — log of diagonal state decay
        B: (b, T, G, N)      — input-to-state projection (SSM B matrix)
        C: (b, T, G, N)      — state-to-output projection (SSM C matrix)
        dt: (b, T, H)        — input-dependent time step (after softplus)
        x: (b, T, H)         — gated input

    Returns:
        y: (b, T, H)         — scan output

    Where b = batch, H = n_heads, N = d_state, G = n_groups.
    Each head belongs to one group: head i -> group (i * G // H).
    """
    batch, seq_len, n_heads = x.shape
    n_groups = B.shape[2]
    heads_per_group = n_heads // n_groups

    # Discretize: A_bar = exp(A * dt), B_bar = B * dt * x
    # A is log-space, so A_bar = exp(A * dt)
    A_bar = jnp.exp(A * dt)  # (B, T, H)

    # Expand B to per-head: (B, T, G, N) -> (B, T, H, N)
    B_expanded = jnp.repeat(B, heads_per_group, axis=2)  # (B, T, H, N)
    C_expanded = jnp.repeat(C, heads_per_group, axis=2)  # (B, T, H, N)

    # B_bar * x: (B, T, H, N) * (B, T, H, 1) -> (B, T, H, N)
    Bu = B_expanded * (dt[..., None] * x[..., None])  # (B, T, H, N)

    # Associative scan operator: each element is (a, b) where
    #   a = A_bar (decay), b = Bu (input contribution)
    # Combine: (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)
    def _binary_op(
        left: Tuple[jax.Array, jax.Array],
        right: Tuple[jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        a_l, b_l = left
        a_r, b_r = right
        return (a_l * a_r, a_r[..., None] * b_l + b_r)

    # A_bar needs shape (B, T, H) for decay, Bu is (B, T, H, N)
    # We need A_bar broadcast: (B, T, H) for the product, (B, T, H, 1) for b update
    decays, states = jax.lax.associative_scan(
        _binary_op,
        (A_bar, Bu),
        axis=1,  # scan along sequence dimension
    )
    # states: (B, T, H, N) — hidden state at each timestep

    # Output: y = C * state, summed over state dim
    y = jnp.sum(states * C_expanded, axis=-1)  # (B, T, H)

    return y


class MambaBlock(Module):
    """Mamba-2 selective state space block.

    Architecture: RMSNorm -> in_proj (gate + x + dt + B + C) ->
    short conv -> SSM -> gate -> out_proj -> residual.
    """

    n_embd: int = field("architecture/n_embd", default=2048)
    n_layers: int = field("architecture/n_layers", default=48)
    d_state: int = field("architecture/d_state", default=128)
    d_conv: int = field("architecture/d_conv", default=4)
    expand: int = field("architecture/expand", default=2)
    n_groups: int = field("architecture/n_groups", default=1)
    n_heads: int = field("architecture/n_heads", default=-1)
    dropout: float = field("architecture/dropout", default=0.0)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        d_inner = self.n_embd * self.expand
        n_heads = self.n_heads if self.n_heads > 0 else d_inner // self.d_state
        self._n_heads = n_heads
        self._d_inner = d_inner

        assert d_inner % n_heads == 0, (
            f"d_inner ({d_inner}) must be divisible by n_heads ({n_heads})"
        )
        assert n_heads % self.n_groups == 0, (
            f"n_heads ({n_heads}) must be divisible by n_groups ({self.n_groups})"
        )

        self.norm = configure(RMSNorm)

        # Project to: gate (d_inner) + x (d_inner) + dt (n_heads) +
        #             B (n_groups * d_state) + C (n_groups * d_state)
        self._proj_size = (
            d_inner  # gate
            + d_inner  # x
            + n_heads  # dt
            + self.n_groups * self.d_state  # B
            + self.n_groups * self.d_state  # C
        )
        init_std = 0.02
        proj_std = 0.02 / math.sqrt(2 * self.n_layers)

        self.in_proj = nn.Dense(
            self._proj_size,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=init_std),
                (Axes.N_EMBD.value, Axes.N_SSM.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        # Short depthwise convolution on x
        self.conv_weight = self.param(
            "conv_weight",
            nn.initializers.normal(stddev=0.02),
            (self.d_conv, d_inner),
            self._param_dtype,
        )
        self.conv_bias = self.param(
            "conv_bias",
            nn.initializers.zeros,
            (d_inner,),
            self._param_dtype,
        )

        # Learnable log(A) for each head — initialized to log of small negative values
        # A controls state decay rate
        self.A_log = self.param(
            "A_log",
            lambda key, shape: jnp.log(
                0.5 + jnp.arange(shape[0], dtype=jnp.float32) * 0.5 / shape[0]
            ),
            (n_heads,),
        )

        # dt bias (learnable, added after projection)
        self.dt_bias = self.param(
            "dt_bias",
            nn.initializers.normal(stddev=0.1),
            (n_heads,),
            self._param_dtype,
        )

        # Output projection
        self.out_proj = nn.Dense(
            self.n_embd,
            use_bias=False,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=proj_std),
                (Axes.N_SSM.value, Axes.N_EMBD.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

    def _short_conv(self, x: jax.Array) -> jax.Array:
        """Apply causal depthwise 1D convolution.

        Args:
            x: (B, T, D) input tensor

        Returns:
            (B, T, D) convolved tensor
        """
        # Pad on the left for causal convolution
        pad_len = self.d_conv - 1
        x_padded = jnp.pad(x, ((0, 0), (pad_len, 0), (0, 0)))

        # Manual depthwise conv: for each position, dot product with kernel
        # conv_weight: (d_conv, d_inner)
        weight = self.conv_weight.astype(x.dtype)
        bias = self.conv_bias.astype(x.dtype)

        # Use lax.conv_general_dilated for efficiency
        # Reshape for grouped convolution: (B, T, D) -> (B, D, T_padded)
        x_t = jnp.transpose(x_padded, (0, 2, 1))  # (B, D, T+pad)
        # Kernel: (d_conv, D) -> (D, 1, d_conv) for depthwise
        w_t = jnp.transpose(weight, (1, 0))[:, None, :]  # (D, 1, d_conv)

        y = jax.lax.conv_general_dilated(
            x_t,
            w_t,
            window_strides=(1,),
            padding="VALID",
            feature_group_count=self._d_inner,
        )  # (B, D, T)
        y = jnp.transpose(y, (0, 2, 1))  # (B, T, D)
        y = y + bias
        return y

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        residual = x
        x = self.norm(x)

        # Project to gate, ssm_input, dt, B, C
        proj = self.in_proj(x)  # (B, T, proj_size)

        d_inner = self._d_inner
        n_heads = self._n_heads
        gs = self.n_groups * self.d_state

        gate = proj[..., :d_inner]
        ssm_in = proj[..., d_inner : 2 * d_inner]
        dt_raw = proj[..., 2 * d_inner : 2 * d_inner + n_heads]
        B = proj[..., 2 * d_inner + n_heads : 2 * d_inner + n_heads + gs]
        C = proj[..., 2 * d_inner + n_heads + gs :]

        # Reshape B, C to (B, T, n_groups, d_state)
        batch, seq_len = x.shape[:2]
        B = B.reshape(batch, seq_len, self.n_groups, self.d_state)
        C = C.reshape(batch, seq_len, self.n_groups, self.d_state)

        # Short convolution + SiLU activation on ssm_in
        ssm_in = self._short_conv(ssm_in)
        ssm_in = jax.nn.silu(ssm_in)

        # dt: softplus to ensure positive, add bias
        dt = jax.nn.softplus(dt_raw + self.dt_bias)  # (B, T, n_heads)

        # A: negative log-space decay (broadcast to (1, 1, n_heads))
        A = -jnp.exp(self.A_log.astype(jnp.float32))  # (n_heads,)
        A = A[None, None, :]  # (1, 1, n_heads)

        # Reshape ssm_in to per-head: (B, T, d_inner) -> (B, T, n_heads)
        # We sum over head_dim to reduce to one scalar per head
        head_dim = d_inner // n_heads
        ssm_in_heads = ssm_in.reshape(batch, seq_len, n_heads, head_dim)

        # Run selective scan in float32 for stability
        A_f32 = jnp.broadcast_to(A, (batch, seq_len, n_heads)).astype(jnp.float32)
        dt_f32 = dt.astype(jnp.float32)
        B_f32 = B.astype(jnp.float32)
        C_f32 = C.astype(jnp.float32)

        # The selective scan operates on (batch, T, n_heads) — one scalar per
        # head.  But our input has head_dim values per head.  We tile the batch
        # dimension by head_dim so each head_dim slice gets its own independent
        # scan, then reshape back.  This avoids vmap (which would prevent XLA
        # from fusing the scans) and keeps a single associative_scan call.
        ssm_flat = ssm_in_heads.transpose(0, 3, 1, 2)  # (B, head_dim, T, n_heads)
        ssm_flat = ssm_flat.reshape(batch * head_dim, seq_len, n_heads)

        A_tiled = jnp.tile(A_f32, (head_dim, 1, 1))  # (B*head_dim, T, H)
        dt_tiled = jnp.tile(dt_f32, (head_dim, 1, 1))  # (B*head_dim, T, H)
        B_tiled = jnp.tile(B_f32, (head_dim, 1, 1, 1))  # (B*head_dim, T, G, N)
        C_tiled = jnp.tile(C_f32, (head_dim, 1, 1, 1))  # (B*head_dim, T, G, N)

        y_flat = _selective_scan(A_tiled, B_tiled, C_tiled, dt_tiled, ssm_flat)
        # y_flat: (B*head_dim, T, n_heads)

        y = y_flat.reshape(batch, head_dim, seq_len, n_heads)
        y = y.transpose(0, 2, 3, 1)  # (B, T, n_heads, head_dim)
        y = y.reshape(batch, seq_len, d_inner)  # (B, T, d_inner)
        y = y.astype(x.dtype)

        # Gate and output
        y = y * jax.nn.silu(gate)

        # Mask padding positions
        if padding_mask is not None:
            y = y * padding_mask[:, :, None].astype(y.dtype)

        y = self.out_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return residual + y
