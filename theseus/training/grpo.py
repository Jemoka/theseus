"""GRPO trainer — PPO with group-relative advantage normalization.

Inherits PPOTrainer; overrides only the per-rollout-reward → per-token-reward
conversion to do group-relative standardization within fixed-size groups.
The flat list of rollouts is reshaped into (n_groups, group_size); within
each group the advantage is (r - mean) / (std + eps).
"""

from dataclasses import dataclass
from typing import Any, List, Type, Generic

import numpy as np
from loguru import logger

from theseus.config import field, configure
from theseus.model.module import Module
from theseus.training.base import M
from theseus.training.ppo import PPOTrainer, BackbonedPPOTrainer


@dataclass
class GRPOConfig:
    group_size: int = field("optimization/grpo/group_size", default=4)


class GRPOTrainer(PPOTrainer[M], Generic[M]):
    """GRPO: PPO with group-relative advantage normalization."""

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return super()._config() + [GRPOConfig]

    def _init_state(self, params: Any) -> None:
        super()._init_state(params)
        self.grpo_config = configure(GRPOConfig)
        if self.main_process():
            logger.info("GRPO | group_size={}", self.grpo_config.group_size)

    def _smear_rewards(
        self, rewards: np.ndarray, action_mask: np.ndarray, discount: float
    ) -> np.ndarray:
        """Group-relative reward smearing.

        Reshapes the per-rollout reward array into (n_groups, group_size),
        standardizes within each group (z-score), then defers to PPO's reward-
        to-go smear for the per-token distribution.

        If the batch size isn't a multiple of group_size, the trailing
        rollouts fall back to plain (un-normalized) reward smearing.
        """
        g = self.grpo_config.group_size
        if g <= 1:
            return super()._smear_rewards(rewards, action_mask, discount)

        B = rewards.shape[0]
        n_full = (B // g) * g
        adv = np.empty_like(rewards)

        if n_full > 0:
            grouped = rewards[:n_full].reshape(-1, g)
            mean = grouped.mean(axis=-1, keepdims=True)
            std = grouped.std(axis=-1, keepdims=True)
            adv[:n_full] = ((grouped - mean) / (std + 1e-8)).reshape(-1)
            logger.debug(
                "GRPO | normalized {} groups of {} (group_mean range [{:.3f},{:.3f}], group_std range [{:.3f},{:.3f}])",
                n_full // g,
                g,
                float(mean.min()),
                float(mean.max()),
                float(std.min()),
                float(std.max()),
            )

        if n_full < B:
            # Trailing rollouts that don't form a full group: keep raw reward.
            adv[n_full:] = rewards[n_full:]
            logger.debug("GRPO | {} trailing rollouts left un-normalized", B - n_full)

        return super()._smear_rewards(adv, action_mask, discount)


class BackbonedGRPOTrainer(BackbonedPPOTrainer, GRPOTrainer[Module]):
    """GRPO trainer that initializes from a pretrained HuggingFace backbone.

    Stacks BackbonedPPOTrainer (HF init + PPO state/forward) with GRPOTrainer
    (group-relative advantage normalization). MRO:
    BackbonedGRPOTrainer → BackbonedPPOTrainer → BackbonedTrainer → GRPOTrainer
    → PPOTrainer → BaseTrainer.
    """

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # super() resolves to BackbonedPPOTrainer, which gives the HF-style
        # config + PPOConfig + RLConfig. Add GRPOConfig on top.
        return super()._config() + [GRPOConfig]
