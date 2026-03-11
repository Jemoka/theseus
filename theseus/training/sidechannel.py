"""
SideChannelTrainer and SideChannelBackboneTrainer for DMA-CoT models.

Custom forward() to handle side-channel batch format:
  {think_x, think_y, sidechannel, sidechannel_mask, padding_mask}

Custom _reshape_batch() and _to_global() to handle the extra N dimension
in the sidechannel tensor.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax.sharding import PartitionSpec as P

from flax.training import train_state

from theseus.base import PyTree, Axis
from theseus.config import field
from theseus.model.module import Module
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.training.flywheel.strategy import Sampling, DatasetStyle


@dataclass
class SideChannelTrainerConfig(BaseTrainerConfig):
    """Config for side-channel training."""

    datasets: List[Sampling] = field(
        "training/dataset",
        default_factory=lambda: [
            Sampling(name="wildchat", rate=1, style=DatasetStyle.SIDECHANNEL)
        ],
    )


def _sidechannel_forward(
    state: train_state.TrainState,
    params: PyTree[jax.Array],
    batch: PyTree[jax.Array],
    key: Optional[jax.Array] = None,
    deterministic: bool = False,
    intermediates: bool = False,
) -> Any:
    """Forward pass for side-channel models.

    Batch contains: think_x, think_y, sidechannel, sidechannel_mask, padding_mask.
    """
    from typing import cast as type_cast

    batch_dict = type_cast(Dict[str, jax.Array], batch)
    x = batch_dict["think_x"]
    y = batch_dict["think_y"]
    padding_mask = batch_dict["padding_mask"]
    sidechannel = batch_dict["sidechannel"]
    sidechannel_mask = batch_dict["sidechannel_mask"]

    dropout_key = None
    if not deterministic and key is not None:
        _, dropout_key = jax_random.split(key)

    rngs = {"dropout": dropout_key} if dropout_key is not None else {}

    if intermediates:
        (logits, loss), mutated = state.apply_fn(
            {"params": params},
            x,
            y,
            padding_mask=padding_mask,
            sidechannel=sidechannel,
            sidechannel_mask=sidechannel_mask,
            deterministic=deterministic,
            rngs=rngs,
            mutable=["intermediates", "plots"],
        )
        return (
            logits,
            loss,
            dict(
                {
                    "intermediates": mutated.get("intermediates", {}),
                    "plots": mutated.get("plots", {}),
                }
            ),
        )
    else:
        logits, loss = state.apply_fn(
            {"params": params},
            x,
            y,
            padding_mask=padding_mask,
            sidechannel=sidechannel,
            sidechannel_mask=sidechannel_mask,
            deterministic=deterministic,
            rngs=rngs,
        )
        return logits, loss, {}


def _sidechannel_reshape_batch(
    self: Any, batch: PyTree[np.ndarray]
) -> PyTree[np.ndarray]:
    """Reshape side-channel batch for sharding.

    Handles the extra N dimension in the sidechannel tensor:
    - 2D arrays (think_x, think_y, padding_mask, sidechannel_mask): reshape to (S, per, L)
    - 3D arrays (sidechannel): reshape to (S, per, N, L)
    """
    from typing import cast as type_cast

    per = self.per_device_batch_size * self.local_replicas

    def _reshape(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            # sidechannel: (total_B, N, L) -> (S, per, N, L)
            return arr.reshape(-1, per, arr.shape[1], arr.shape[2])
        else:
            return arr.reshape(-1, per, arr.shape[-1])

    return type_cast(PyTree[np.ndarray], jax.tree_util.tree_map(_reshape, batch))


def _sidechannel_to_global(
    self: Any, batch: PyTree[np.ndarray]
) -> PyTree[jax.Array]:
    """Move batch to global arrays, handling sidechannel's extra dimension."""
    from typing import cast as type_cast
    from jax.experimental import multihost_utils

    pspec_2d = P(None, Axis.BATCH, None)  # type: ignore
    pspec_3d = P(None, Axis.BATCH, None, None)  # type: ignore

    def convert(arr: np.ndarray) -> jax.Array:
        spec = pspec_3d if arr.ndim == 4 else pspec_2d
        return multihost_utils.host_local_array_to_global_array(  # type: ignore[no-any-return]
            arr, self.mesh, spec
        )

    return type_cast(PyTree[jax.Array], jax.tree_util.tree_map(convert, batch))


class SideChannelTrainer(BaseTrainer[SideChannelTrainerConfig, Module]):
    """Trainer for DMA-CoT models with side-channel batches.

    Use for training from scratch (GPT pretrain stage 1 & finetune stage 2).
    """

    MODEL: Type[Module] = Module  # type: ignore[type-abstract]
    CONFIG = SideChannelTrainerConfig

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        return _sidechannel_forward(
            state, params, batch, key, deterministic, intermediates
        )

    def _reshape_batch(self, batch: PyTree[np.ndarray]) -> PyTree[np.ndarray]:
        return _sidechannel_reshape_batch(self, batch)

    def _to_global(self, batch: PyTree[np.ndarray]) -> PyTree[jax.Array]:
        return _sidechannel_to_global(self, batch)


class SideChannelBackboneTrainer(BackbonedTrainer):
    """Trainer that loads pretrained HF backbone and adds cross-attention side channels.

    Inherits BackbonedTrainer for HF weight loading, overrides forward()
    for side-channel batch format.
    """

    CONFIG = SideChannelTrainerConfig

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        return _sidechannel_forward(
            state, params, batch, key, deterministic, intermediates
        )

    def _reshape_batch(self, batch: PyTree[np.ndarray]) -> PyTree[np.ndarray]:
        return _sidechannel_reshape_batch(self, batch)

    def _to_global(self, batch: PyTree[np.ndarray]) -> PyTree[jax.Array]:
        return _sidechannel_to_global(self, batch)
