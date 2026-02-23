from dataclasses import dataclass
from typing import cast as type_cast
from typing import Any, Dict, Optional, List, Type, Generic

import jax
import jax.numpy as jnp
import jax.random as jax_random
import flax
import flax.linen
from flax.training import train_state

import optax
from loguru import logger

from theseus.base import PyTree
from theseus.model.module import Module
from theseus.config import field, configure
from theseus.training.trainer import BaseTrainer, BaseTrainerConfig, M
from theseus.training.backbone import BackbonedTrainer


@dataclass
class DPOConfig:
    beta: float = field("optimization/dpo/beta", default=0.1)
    label_smoothing: float = field("optimization/dpo/label_smoothing", default=0.0)


class ContrastiveTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    base: PyTree[Any]
    beta: float
    label_smooth: float


class ContrastiveTrainer(BaseTrainer[BaseTrainerConfig, M], Generic[M]):
    """Contrastive trainer scaffold."""

    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return BaseTrainer._config() + [DPOConfig]

    def _init_optimizer(self, params: PyTree[jax.Array]) -> None:
        """Build optimizer, scheduler, and sharded contrastive train state."""

        # initialize DPO configuration from state, we'll use it in loss computation
        self.dpo_config = configure(DPOConfig)

        # build the optimizer
        self.scheduler: optax._src.base.Schedule = self._schedule()
        self.tx = self._optimizer()

        # ContrastiveTrainState has a different pytree structure than TrainState:
        # it includes `base` (same shape as params) plus scalar `beta`/`label_smooth`.
        # eval_shape traces the full state creation to get correct sharding for all fields.
        label_smooth = self.dpo_config.label_smoothing
        beta = self.dpo_config.beta

        def make_state(p: PyTree[jax.Array]) -> ContrastiveTrainState:
            return type_cast(
                ContrastiveTrainState,
                ContrastiveTrainState.create(
                    apply_fn=self.model.apply,
                    params=p,
                    base=p,  # type: ignore
                    tx=self.tx,
                    label_smooth=label_smooth,
                    beta=beta,
                ),
            )

        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
            flax.linen.get_partition_spec(state_shapes),
            self.mesh,
            rules=tuple(self.model.sharding),
        )
        self.state = jax.jit(make_state, out_shardings=self.state_sharding)(params)

        self.total_params = (
            sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        )

        if self.main_process():
            logger.info(f"MODEL | Total Parameters: {self.total_params:.2f}m")

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        mutable: Optional[list[str]] = None,
        extra_variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        cstate = type_cast(ContrastiveTrainState, state)
        batch_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], batch)

        # in this case we are probably running some kind of
        # evaluation so we'd like to skip the contrastive loss and
        # just return the logits
        if batch_dict.get("x") is not None:
            return BaseTrainer.forward(
                state,
                params,
                batch,
                key,
                deterministic,
                mutable,
                extra_variables,
            )

        # if mutable is not None, panic. something is likely goofy
        # such as we are trying to do KV caching during training?
        if mutable is not None:
            raise NotImplementedError(
                "Mutable variables not supported in contrastive trainer yet."
            )

        # unpack dataset
        pos = batch_dict["pos"]
        neg = batch_dict["neg"]
        padding_mask_pos = batch_dict["padding_mask_pos"]
        padding_mask_neg = batch_dict["padding_mask_neg"]

        # build dropout details / extra variables
        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)
        kwargs: Dict[str, Any] = {
            "deterministic": deterministic,
        }
        if dropout_key is not None:
            kwargs["rngs"] = {"dropout": dropout_key}

        # compute logprobs for pos and neg using the parameters
        (logits, loss_pos) = cstate.apply_fn(
            {"params": params},
            pos[:, :-1],
            pos[:, 1:],
            padding_mask=padding_mask_pos[:, :-1],
            **kwargs,
        )
        (_, loss_neg) = cstate.apply_fn(
            {"params": params},
            neg[:, :-1],
            neg[:, 1:],
            padding_mask=padding_mask_neg[:, :-1],
            **kwargs,
        )

        # compute baseline loss
        (_, loss_base_pos) = cstate.apply_fn(
            {"params": cstate.base},
            pos[:, :-1],
            pos[:, 1:],
            padding_mask=padding_mask_pos[:, :-1],
            **kwargs,
        )
        (_, loss_base_neg) = cstate.apply_fn(
            {"params": cstate.base},
            neg[:, :-1],
            neg[:, 1:],
            padding_mask=padding_mask_neg[:, :-1],
            **kwargs,
        )
        # detach
        loss_base_pos = jax.lax.stop_gradient(loss_base_pos)
        loss_base_neg = jax.lax.stop_gradient(loss_base_neg)

        # compute DPO loss and rewards
        logits = -((loss_pos - loss_neg) - (loss_base_pos - loss_base_neg))

        beta = cstate.beta
        label_smooth = cstate.label_smooth

        loss = (
            -jax.nn.log_sigmoid(beta * logits) * (1 - label_smooth)
            - jax.nn.log_sigmoid(-beta * logits) * label_smooth
        )

        # reward sign fix
        chosen_rewards = jax.lax.stop_gradient(-beta * (loss_pos - loss_base_pos))
        rejected_rewards = jax.lax.stop_gradient(-beta * (loss_neg - loss_base_neg))

        metrics = {
            "rewards/chosen": chosen_rewards,
            "rewards/rejected": rejected_rewards,
            "rewards/reward_accuracy": jnp.mean(chosen_rewards > rejected_rewards),
            "rewards/reward_margin": chosen_rewards - rejected_rewards,
            "policy/nll_chosen": loss_pos,
            "policy/nll_rejected": loss_neg,
            "ref/nll_chosen": loss_base_pos,
            "ref/nll_rejected": loss_base_neg,
        }

        return logits, loss, metrics


class BackbonedContrastiveTrainer(BackbonedTrainer, ContrastiveTrainer[Module]):
    """Contrastive trainer scaffold for backboned training."""

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return BackbonedTrainer._config() + [DPOConfig]
