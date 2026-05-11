"""Mamba-2 selective state space block.

Implements the Mamba-2 architecture (Dao & Gu, 2024) via its **State Space
Duality (SSD)** chunked-matmul algorithm: the recurrence is split along T
into chunks of size ``Q``, with attention-style dense matmuls inside each
chunk and a much shorter associative scan over chunk-boundary states.  This
keeps activation memory at ``O(B·T·H·N/Q + B·n_chunks·H·N·P)`` and turns
the inner loop into tensor-core-friendly GEMMs.
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


def _ssd_scan(
    A: jax.Array,        # (B, T, H) — log of diagonal state decay (negative)
    B: jax.Array,        # (B, T, G, N) — SSM input projection
    C: jax.Array,        # (B, T, G, N) — SSM output projection
    dt: jax.Array,       # (B, T, H) — input-dependent time step (post softplus)
    x: jax.Array,        # (B, T, H, P) — gated input, P = head_dim channels
    *,
    chunk_size: int = 64,
) -> jax.Array:
    """Mamba-2 SSD (State Space Duality) chunked-matmul selective scan.

    Implements the recurrence
        state[t] = exp(A[t] * dt[t]) * state[t-1] + B[t] * dt[t] * x[t]
        y[t]     = C[t] · state[t]
    where ``state`` lives in R^N per head and ``x`` has ``P = head_dim``
    independent channels per head that share ``A``, ``B``, ``C``, ``dt``.

    The recurrence is computed in two halves:

    * **Intra-chunk** — for each chunk of length ``Q``, build the (Q, Q)
      lower-triangular decay matrix ``L[i, j] = prod_{k=j+1..i} a[k]`` and
      compute ``y_intra = einsum(L · ⟨C, B⟩ · dt, x)``.  Pure GEMMs.
    * **Inter-chunk** — accumulate each chunk's input-driven state
      contribution and decay across chunk boundaries via a length-``T/Q``
      ``jax.lax.associative_scan``, then add the carryover to ``y`` inside
      each chunk via another einsum with ``C``.

    Args:
        A:  ``(B, T, H)`` log decay per head; expected negative.
        B:  ``(B, T, G, N)`` input projection (``G`` groups share across
            ``n_heads // G`` heads each).
        C:  ``(B, T, G, N)`` output projection.
        dt: ``(B, T, H)`` per-head time step (must be ≥ 0).
        x:  ``(B, T, H, P)`` gated input channels.

    Keyword Args:
        chunk_size: ``Q``. ``T`` is zero-padded to a multiple of this if
            necessary; padded positions contribute nothing because ``dt``
            and ``x`` are zero there.  Standard choices are 64 or 128.

    Returns:
        ``y`` of shape ``(B, T, H, P)``.
    """
    batch, seq_len, n_heads, P = x.shape
    n_groups = B.shape[2]
    N = B.shape[3]
    assert n_heads % n_groups == 0, (
        f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})"
    )
    Hg = n_heads // n_groups  # heads per group
    G = n_groups
    Q = chunk_size

    # Pad T up to a multiple of Q.  dt and x are zero-padded so padded
    # positions contribute nothing to either the within-chunk matmul or the
    # chunk-final state (the trailing outputs are sliced off at the end).
    pad_len = (-seq_len) % Q
    if pad_len:
        A = jnp.pad(A, ((0, 0), (0, pad_len), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        C = jnp.pad(C, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    T_pad = seq_len + pad_len
    n_chunks = T_pad // Q

    A_c = A.reshape(batch, n_chunks, Q, n_heads)
    B_c = B.reshape(batch, n_chunks, Q, G, N)
    C_c = C.reshape(batch, n_chunks, Q, G, N)
    dt_c = dt.reshape(batch, n_chunks, Q, n_heads)
    x_c = x.reshape(batch, n_chunks, Q, n_heads, P)

    # Cumulative log decay within each chunk (inclusive).
    log_a = A_c * dt_c                          # (B, c, Q, H)
    cum_log_a = jnp.cumsum(log_a, axis=2)       # (B, c, Q, H), sum_{k=0..i}
    chunk_log_decay = cum_log_a[:, :, -1, :]    # (B, c, H), full-chunk decay

    # Causal decay matrix
    #   L[c, i, j, h] = exp(cum_log_a[c, i, h] - cum_log_a[c, j, h])  for j ≤ i
    #                = 0                                              otherwise
    # We mask the upper triangle in log-space before exp so we never compute
    # exp of a large positive number (which would overflow in float32 for
    # long chunks).
    log_diff = cum_log_a[:, :, :, None, :] - cum_log_a[:, :, None, :, :]
    causal = jnp.tril(jnp.ones((Q, Q), dtype=jnp.bool_))[None, None, :, :, None]
    L = jnp.exp(jnp.where(causal, log_diff, -jnp.inf))  # (B, c, Q, Q, H)

    # ---- Intra-chunk ---------------------------------------------------
    # CB[c, i, j, g] = ⟨C[c, i, g, :], B[c, j, g, :]⟩
    CB = jnp.einsum("bcign,bcjgn->bcijg", C_c, B_c)  # (B, c, Q, Q, G)

    # Reshape the head axis as (G, Hg) so heads inherit their group's B/C.
    L_rs = L.reshape(batch, n_chunks, Q, Q, G, Hg)
    dt_rs = dt_c.reshape(batch, n_chunks, Q, G, Hg)
    x_rs = x_c.reshape(batch, n_chunks, Q, G, Hg, P)

    # y_intra[c, i, g, h, p] = sum_j L[c, i, j, g, h] · CB[c, i, j, g]
    #                                · dt[c, j, g, h] · x[c, j, g, h, p]
    y_intra = jnp.einsum(
        "bcijGH,bcijG,bcjGH,bcjGHp->bciGHp",
        L_rs, CB, dt_rs, x_rs,
    )

    # ---- Per-chunk state contribution ---------------------------------
    # decay_to_end[c, j, h] = prod_{k=j+1..Q-1} a[c, k, h]
    #                      = exp(chunk_log_decay[c, h] - cum_log_a[c, j, h])
    decay_to_end = jnp.exp(chunk_log_decay[:, :, None, :] - cum_log_a)
    decay_to_end_rs = decay_to_end.reshape(batch, n_chunks, Q, G, Hg)

    # state_local[c, g, h, n, p]
    #   = sum_j decay_to_end[c, j, g, h] · dt[c, j, g, h]
    #           · B[c, j, g, n] · x[c, j, g, h, p]
    state_local = jnp.einsum(
        "bcjGH,bcjGH,bcjGn,bcjGHp->bcGHnp",
        decay_to_end_rs, dt_rs, B_c, x_rs,
    )

    # ---- Inter-chunk associative scan over chunk states ---------------
    # Linear recurrence over chunks:
    #   state[c] = chunk_decay[c] * state[c-1] + state_local[c],  state[-1] = 0.
    chunk_decay = jnp.exp(chunk_log_decay).reshape(batch, n_chunks, G, Hg)

    def scan_op(
        left: Tuple[jax.Array, jax.Array],
        right: Tuple[jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        d_l, s_l = left
        d_r, s_r = right
        # combine(left, right): "left happens first, right after".
        return (d_l * d_r, d_r[..., None, None] * s_l + s_r)

    _, state_at_end = jax.lax.associative_scan(
        scan_op, (chunk_decay, state_local), axis=1
    )  # state_at_end[c] is the state after processing chunk c.

    # state_at_start[c] = state_at_end[c-1]; state_at_start[0] = 0.
    state_at_start = jnp.concatenate(
        [jnp.zeros_like(state_at_end[:, :1]), state_at_end[:, :-1]],
        axis=1,
    )

    # ---- Inter-chunk contribution to y --------------------------------
    # decay_from_start[c, i, h] = prod_{k=0..i} a[c, k, h] = exp(cum_log_a)
    decay_from_start = jnp.exp(cum_log_a).reshape(batch, n_chunks, Q, G, Hg)

    # y_inter[c, i, g, h, p] = decay_from_start[c, i, g, h]
    #                          · sum_n C[c, i, g, n] · state_at_start[c, g, h, n, p]
    y_inter = jnp.einsum(
        "bciGH,bciGn,bcGHnp->bciGHp",
        decay_from_start, C_c, state_at_start,
    )

    # ---- Combine, reshape, slice off padding --------------------------
    y = (y_intra + y_inter).reshape(batch, T_pad, n_heads, P)
    if pad_len:
        y = y[:, :seq_len]
    return y


def _selective_scan(
    A: jax.Array,        # (B, T, H)
    B: jax.Array,        # (B, T, G, N)
    C: jax.Array,        # (B, T, G, N)
    dt: jax.Array,       # (B, T, H)
    x: jax.Array,        # (B, T, H)
    *,
    chunk_size: int = 64,
) -> jax.Array:
    """Single-channel wrapper around :func:`_ssd_scan`.

    Accepts scalar-per-head ``x`` of shape ``(B, T, H)`` (the historic
    Mamba-1 / Mamba-Triton signature), adds a unit channel dim, runs the
    SSD scan, and squeezes it off.  Kept as a stable public surface for
    tests and any external callers; new code should call ``_ssd_scan``
    directly with multi-channel ``x``.
    """
    return _ssd_scan(A, B, C, dt, x[..., None], chunk_size=chunk_size)[..., 0]


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
        self.conv_weight: jax.Array = self.param(
            "conv_weight",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (None, Axes.N_SSM.value),
            ),
            (self.d_conv, d_inner),
            self._param_dtype,
        )  # type: ignore
        self.conv_bias: jax.Array = self.param(
            "conv_bias",
            nn.with_partitioning(
                nn.initializers.zeros,
                (Axes.N_SSM.value,),
            ),
            (d_inner,),
            self._param_dtype,
        )  # type: ignore

        # Learnable log(A) for each head — initialized to log of small negative values
        # A controls state decay rate; stored as log|A| so exp is always positive
        self.A_log = self.param(
            "A_log",
            lambda key, shape: jnp.log(jnp.arange(1, shape[0] + 1, dtype=jnp.float32)),
            (n_heads,),
        )

        # dt bias (learnable, added after projection)
        self.dt_bias = self.param(
            "dt_bias",
            nn.initializers.normal(stddev=0.1),
            (n_heads,),
            self._param_dtype,
        )

        # D: direct skip connection from conv output to SSM output (one per channel)
        self.D: jax.Array = self.param(
            "D",
            nn.with_partitioning(
                nn.initializers.ones,
                (Axes.N_SSM.value,),
            ),
            (d_inner,),
            self._param_dtype,
        )  # type: ignore

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

        # Per-head channels: (B, T, d_inner) -> (B, T, n_heads, head_dim).
        # SSD treats head_dim as independent channels that share A/B/C/dt
        # within a head — no batch-axis tiling required.
        head_dim = d_inner // n_heads
        ssm_in_heads = ssm_in.reshape(batch, seq_len, n_heads, head_dim)

        # Run the SSD selective scan in float32 for stability.
        A_f32 = jnp.broadcast_to(A, (batch, seq_len, n_heads)).astype(jnp.float32)
        dt_f32 = dt.astype(jnp.float32)
        B_f32 = B.astype(jnp.float32)
        C_f32 = C.astype(jnp.float32)
        x_f32 = ssm_in_heads.astype(jnp.float32)

        y = _ssd_scan(A_f32, B_f32, C_f32, dt_f32, x_f32)  # (B, T, H, head_dim)
        y = y.reshape(batch, seq_len, d_inner)
        y = y.astype(x.dtype)

        # D skip connection: direct path from conv output to SSM output
        y = y + self.D.astype(y.dtype) * ssm_in

        # Gate and output
        y = y * jax.nn.silu(gate)

        # Mask padding positions
        if padding_mask is not None:
            y = y * padding_mask[:, :, None].astype(y.dtype)

        y = self.out_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return residual + y
