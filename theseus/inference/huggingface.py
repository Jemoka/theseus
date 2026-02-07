"""
HuggingFace-specific inference job helpers.
"""

from typing import Any, Generic, Optional, Tuple, TypeVar, cast

import flax
import jax
import jax.numpy as jnp
from flax.training import train_state

from theseus.inference.base import InferenceJob
from theseus.model.module import Module

C = TypeVar("C")
M = TypeVar("M", bound=Module)


class HFInferenceState(train_state.TrainState):  # type: ignore[no-untyped-call]
    buffers: Any


class HFInferenceJob(InferenceJob[C, M], Generic[C, M]):
    @staticmethod
    def _init_template_state(model: M, block_size: int) -> train_state.TrainState:
        import optax

        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, block_size), dtype=jnp.int32)
        variables = model.init(key, dummy_input)
        params = variables["params"]
        buffers = variables.get("buffers", flax.core.freeze({}))
        state = HFInferenceState.create(  # type: ignore[no-untyped-call]
            apply_fn=model.apply,
            params=params,
            tx=optax.identity(),
            buffers=buffers,
        )
        return cast(train_state.TrainState, state)

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: Any,
        batch: Tuple[jax.Array, Optional[jax.Array], jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        x, y, padding_mask = batch
        del key, deterministic

        buffers = getattr(state, "buffers", None)
        variables = {"params": params}
        if buffers is not None:
            variables["buffers"] = buffers

        logits, loss = state.apply_fn(
            variables,
            x,
            y,
            padding_mask=padding_mask,
        )
        return logits, loss
