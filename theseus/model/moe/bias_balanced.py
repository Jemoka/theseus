"""Bias-balanced MoE routing."""

from typing import Tuple

import jax
import jax.numpy as jnp

from theseus.config import field
from theseus.model.moe.base import MoE


class BiasBalancedMoE(MoE):
    """Batch-local approximation of DeepSeek-style loss-free balancing.

    DeepSeek updates a persistent expert bias using recent routing load. To avoid
    threading mutable model state throughout the repo, this implementation uses
    the current batch's observed load to compute a one-step balancing bias and
    reroutes once with that bias applied.
    """

    # In this stateless variant, bias_update_rate is the maximum absolute
    # correction applied to any expert during the reroute pass.
    bias_update_rate: float = field("architecture/moe/bias_update_rate", default=0.25)
    bias_smoothing: float = field("architecture/moe/bias_smoothing", default=1.0)

    def _validate_config(self) -> None:
        super()._validate_config()
        if self.bias_update_rate < 0:
            raise ValueError("architecture/moe/bias_update_rate must be >= 0")
        if self.bias_smoothing <= 0:
            raise ValueError("architecture/moe/bias_smoothing must be > 0")

    def _select_experts(self, router_logits: jax.Array) -> Tuple[jax.Array, jax.Array]:
        k = min(self.k, self.num_experts)

        _, initial_idx = jax.lax.top_k(router_logits, k=k)
        counts = jnp.bincount(initial_idx.reshape(-1), length=self.num_experts)
        target = router_logits.shape[0] * k / self.num_experts

        # Normalize by expected load and pass through tanh so the correction is
        # smooth, bounded, and cannot overwhelm the learned router.
        load_error = (counts.astype(router_logits.dtype) - target) / max(target, 1.0)
        balance_bias = -self.bias_update_rate * jnp.tanh(
            load_error / self.bias_smoothing
        )
        balance_bias = jax.lax.stop_gradient(balance_bias)
        _, expert_idx = jax.lax.top_k(router_logits + balance_bias[None, :], k=k)

        selected_logits = jnp.take_along_axis(router_logits, expert_idx, axis=-1)
        weights = jax.nn.softmax(selected_logits, axis=-1)
        return weights, expert_idx
