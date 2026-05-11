"""LaCT (Large-Chunk Test-Time Training) block.

Three residual sublayers in sequence:

1. Sliding-window attention (``GroupedSelfAttention`` with ``sliding=True``) —
   the local mixer that re-establishes intra-chunk order/causality the TTT
   layer deliberately ignores.
2. The TTT layer — Q/K/V projections + per-token eta predictor + a SwiGLU
   "fast-weight" network ``f_W`` whose weights ``W`` are updated once per
   chunk of ``b`` tokens via gradient descent on ``L(W) = -Σ eta · <f_W(k), v>``,
   then queried as ``f_W(q)``.  The initial fast weights ``W*_0`` are slow
   params (learned by outer SGD); at inference, a mutable ``"fast_weights"``
   collection persists the post-update ``W`` across forward calls.
3. A standard feed-forward MLP.

The TTT layer treats the chunk as an unordered set — that is the paper's whole
point.  Causality across chunks comes from the ``apply_then_update`` execution
order (query a chunk before writing its keys/values into ``W``), no masking
machinery required.
"""

from typing import Any, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import configure, field
from theseus.model.attention.grouped import GroupedSelfAttention
from theseus.model.axes import Axes
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.layers.mlp import MLP
from theseus.model.layers.lact import (
    FastMomentum,
    FastWeights,
    batch_broadcast,
    batch_zeros_momentum,
    chunked_update_and_apply,
)
from theseus.model.module import Module


class LaCTBlock(Module):
    n_embd: int = field("architecture/n_embd", default=2048)
    n_layers: int = field("architecture/n_layers", default=16)
    dropout: float = field("architecture/dropout", default=0.0)
    bias: bool = field("architecture/bias", default=True)

    # Fast-weight network shape
    fw_inter_size: int = field(
        "architecture/fw_inter_size", default=-1
    )  # -1 → fall back to n_embd

    # Inner-loop hyperparameters
    chunk_size: int = field("architecture/ttt_chunk_size", default=2048)
    ttt_optimizer: str = field("architecture/ttt_optimizer", default="muon")
    ttt_momentum: float = field("architecture/ttt_momentum", default=0.9)
    apply_then_update: bool = field("architecture/ttt_apply_then_update", default=True)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, GroupedSelfAttention, MLP]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    @property
    def _fw_h(self) -> int:
        return self.fw_inter_size if self.fw_inter_size > 0 else self.n_embd

    def setup(self) -> None:
        h, d = self._fw_h, self.n_embd

        self.ln_a = configure(LayerNorm)
        self.swa = configure(GroupedSelfAttention)
        self.ln_b = configure(LayerNorm)

        # TTT layer projections — single-head; Q/K/V live at model dim.
        # Output axis is None (replicated) so the kernel doesn't repeat the
        # N_EMBD logical name on both dims (Flax forbids that), and so the
        # contraction inside ``apply_fw`` with W1/W2/W3 (which use N_EMBD on
        # their contracted axis) doesn't require an all-gather to align.
        qkv_axes = (Axes.N_EMBD.value, None)
        self.q_proj = nn.Dense(
            d,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02), qkv_axes
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.k_proj = nn.Dense(
            d,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02), qkv_axes
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.v_proj = nn.Dense(
            d,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02), qkv_axes
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.eta_head = nn.Dense(
            1,
            use_bias=True,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.zeros, (Axes.N_EMBD.value, None)
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        # Initial fast weights — slow params, learned by outer SGD.
        self.W1_0 = self.param(
            "W1_0",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.N_FW.value, Axes.N_EMBD.value),
            ),
            (h, d),
            self._param_dtype,
        )
        self.W2_0 = self.param(
            "W2_0",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_FW.value),
            ),
            (d, h),
            self._param_dtype,
        )
        self.W3_0 = self.param(
            "W3_0",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.N_FW.value, Axes.N_EMBD.value),
            ),
            (h, d),
            self._param_dtype,
        )

        self.out_proj = nn.Dense(
            d,
            use_bias=self.bias,
            kernel_init=nn.with_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (None, Axes.N_EMBD.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        self.ln_c = configure(LayerNorm)
        self.mlp = configure(MLP)

    # ------------------------------------------------------------------
    # TTT sublayer
    # ------------------------------------------------------------------

    def _initial_fw(self, batch_size: int, dtype: Any) -> FastWeights:
        """Broadcast the slow ``W*_0`` to a batched FastWeights pytree."""
        W_single = FastWeights(
            jnp.asarray(self.W1_0, dtype),
            jnp.asarray(self.W2_0, dtype),
            jnp.asarray(self.W3_0, dtype),
        )
        return batch_broadcast(W_single, batch_size)

    @nn.compact
    def _ttt(self, x: jax.Array, padding_mask: Optional[jax.Array]) -> jax.Array:
        B, T, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Sigmoid keeps eta ≥ 0 so the inner loss can only decrease the running W;
        # paper sets eta as an input-dependent scalar per token.
        eta = jax.nn.sigmoid(self.eta_head(x).astype(jnp.float32))[..., 0]
        if padding_mask is not None:
            eta = eta * padding_mask.astype(eta.dtype)

        compute_dtype = jnp.float32  # inner GD is numerically delicate; run f32
        Q_f = Q.astype(compute_dtype)
        K_f = K.astype(compute_dtype)
        V_f = V.astype(compute_dtype)
        W0_batched = self._initial_fw(B, compute_dtype)
        M0_batched = batch_zeros_momentum(W0_batched)

        is_inference = self.is_mutable_collection("fast_weights")

        if is_inference:
            # Variables seed from the CURRENT W*_0 the first time this collection
            # is materialized. When the caller does not pass a previously-mutated
            # fast_weights collection back as extra_variables, init_fn runs again
            # against the current slow params — this is the W-emptying mechanism.
            W_var = self.variable("fast_weights", "W", lambda: W0_batched)
            M_var = self.variable("fast_weights", "M", lambda: M0_batched)
            W_in: FastWeights = W_var.value
            M_in: FastMomentum = M_var.value
        else:
            W_in = W0_batched
            M_in = M0_batched

        if is_inference and T < self.chunk_size:
            # Decode step: apply current W without firing the inner update —
            # a single-token chunk would otherwise overwrite the prefill-learned
            # W with a degenerate gradient on one token.
            out_f = jnp.einsum("bhd,btd->bth", W_in.W1, Q_f)
            out_f = jax.nn.silu(out_f) * jnp.einsum("bhd,btd->bth", W_in.W3, Q_f)
            out_f = jnp.einsum("bdh,bth->btd", W_in.W2, out_f)
            W_final, M_final = W_in, M_in
        else:
            out_f, W_final, M_final = chunked_update_and_apply(
                W_in,
                M_in,
                K_f,
                V_f,
                eta.astype(compute_dtype),
                Q_f,
                self.chunk_size,
                self.ttt_optimizer,
                self.ttt_momentum,
                self.apply_then_update,
            )

        if is_inference:
            W_var.value = W_final
            M_var.value = M_final

        out = out_f.astype(self._activation_dtype)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Block forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        # Sublayer 1: SWA. `sliding=True` enables the sliding-window mask;
        # GroupedSelfAttention falls back to causal if its sliding_window
        # field is ≤ 0, so this is safe even when SWA is disabled by config.
        h_swa = self.swa(
            self.ln_a(x),
            padding_mask=padding_mask,
            deterministic=deterministic,
            sliding=True,
            **kwargs,
        )
        x = x + h_swa

        # Sublayer 2: TTT
        h_ttt = self._ttt(self.ln_b(x), padding_mask=padding_mask)
        x = x + h_ttt

        # Sublayer 3: FFN
        h_mlp = self.mlp(self.ln_c(x), deterministic=deterministic)
        x = x + h_mlp
        return x
