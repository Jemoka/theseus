"""Mixture-of-experts feed-forward modules."""

import math
from typing import Any, List, Optional, Tuple, Type

import flax.linen as nn
import jax
import jax.numpy as jnp

from theseus.config import configure, field
from theseus.model.layers.mlp import MLP
from theseus.model.module import Module


class MoE(Module):
    """Base MoE feed-forward layer with fixed-capacity expert packing."""

    num_experts: int = field("architecture/moe/experts", default=4)
    k: int = field("architecture/moe/experts_per_embd", default=1)
    capacity_factor: float = field("architecture/moe/capacity_factor", default=1.0)
    capacity_round_to: int = field("architecture/moe/capacity_round_to", default=128)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [MLP]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        self._validate_config()

        ExpertMLP = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(0, None),
            out_axes=0,
            axis_size=self.num_experts,
            metadata_params={"partition_name": None},
        )
        self.experts = configure(ExpertMLP)
        self.router = nn.Dense(
            self.num_experts,
            use_bias=False,
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )

    def _validate_config(self) -> None:
        if self.k < 1:
            raise ValueError("architecture/moe/experts_per_embd must be >= 1")
        if self.num_experts < 1:
            raise ValueError("architecture/moe/experts must be >= 1")
        if self.capacity_factor <= 0:
            raise ValueError("architecture/moe/capacity_factor must be > 0")
        if self.capacity_round_to < 1:
            raise ValueError("architecture/moe/capacity_round_to must be >= 1")

    def _capacity(self, num_tokens: int) -> int:
        capacity = math.ceil(self.capacity_factor * num_tokens)
        round_to = self.capacity_round_to
        capacity = round_to * math.ceil(capacity / round_to)
        return max(1, min(capacity, num_tokens))

    def _router_logits(self, flat_x: jax.Array) -> jax.Array:
        return self.router(flat_x).astype(jnp.float32)

    def _select_experts(self, router_logits: jax.Array) -> Tuple[jax.Array, jax.Array]:
        k = min(self.k, self.num_experts)
        _, expert_idx = jax.lax.top_k(router_logits, k=k)
        selected_logits = jnp.take_along_axis(router_logits, expert_idx, axis=-1)
        weights = nn.softmax(selected_logits, axis=-1)
        return weights, expert_idx

    def _pack_assignments(
        self,
        flat_x: jax.Array,
        weights: jax.Array,
        expert_idx: jax.Array,
        capacity: int,
    ) -> Tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        num_tokens, hidden_size = flat_x.shape

        flat_expert_idx = expert_idx.reshape(-1).astype(jnp.int32)
        flat_weights = weights.reshape(-1)
        token_idx = jnp.broadcast_to(
            jnp.arange(num_tokens, dtype=jnp.int32)[:, None],
            expert_idx.shape,
        ).reshape(-1)

        # Group assignments by expert so we can build a single packed expert
        # buffer and run one vmapped expert stack over it.
        order = jnp.argsort(flat_expert_idx, stable=True)
        sorted_expert_idx = flat_expert_idx[order]
        sorted_token_idx = token_idx[order]
        sorted_weights = flat_weights[order]
        sorted_x = flat_x[sorted_token_idx]

        counts = jnp.bincount(sorted_expert_idx, length=self.num_experts)
        expert_offsets = jnp.cumsum(counts, dtype=jnp.int32) - counts
        slot_idx = (
            jnp.arange(sorted_expert_idx.shape[0], dtype=jnp.int32)
            - expert_offsets[sorted_expert_idx]
        )
        keep = slot_idx < capacity

        kept_expert_idx = sorted_expert_idx[keep]
        kept_slot_idx = slot_idx[keep]
        kept_token_idx = sorted_token_idx[keep]
        kept_weights = sorted_weights[keep]
        kept_x = sorted_x[keep]

        expert_inputs = jnp.zeros(
            (self.num_experts, capacity, hidden_size),
            dtype=flat_x.dtype,
        )
        expert_inputs = expert_inputs.at[kept_expert_idx, kept_slot_idx].set(kept_x)

        clipped_counts = jnp.minimum(counts, capacity)
        valid_slots = (
            jnp.arange(capacity, dtype=jnp.int32)[None, :] < clipped_counts[:, None]
        )[..., None]

        return (
            expert_inputs,
            valid_slots,
            kept_expert_idx,
            kept_slot_idx,
            kept_token_idx,
            kept_weights,
        )

    def _combine_expert_outputs(
        self,
        expert_outputs: jax.Array,
        kept_expert_idx: jax.Array,
        kept_slot_idx: jax.Array,
        kept_token_idx: jax.Array,
        kept_weights: jax.Array,
        num_tokens: int,
    ) -> jax.Array:
        kept_outputs = expert_outputs[kept_expert_idx, kept_slot_idx]
        weighted_outputs = kept_outputs * kept_weights[:, None].astype(
            kept_outputs.dtype
        )
        combined = jnp.zeros(
            (num_tokens, expert_outputs.shape[-1]),
            dtype=weighted_outputs.dtype,
        )
        return combined.at[kept_token_idx].add(weighted_outputs)

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        """Apply top-k expert routing to ``x`` of shape ``[B, T, H]``."""

        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        flat_x = x.reshape(num_tokens, hidden_size)

        router_logits = self._router_logits(flat_x)
        weights, expert_idx = self._select_experts(router_logits)

        (
            expert_inputs,
            valid_slots,
            kept_expert_idx,
            kept_slot_idx,
            kept_token_idx,
            kept_weights,
        ) = self._pack_assignments(
            flat_x,
            weights,
            expert_idx,
            capacity=self._capacity(num_tokens),
        )

        expert_outputs = self.experts(expert_inputs, deterministic)
        expert_outputs = jnp.where(valid_slots, expert_outputs, 0)
        combined = self._combine_expert_outputs(
            expert_outputs,
            kept_expert_idx,
            kept_slot_idx,
            kept_token_idx,
            kept_weights,
            num_tokens,
        )
        return combined.reshape(batch_size, seq_len, hidden_size)
