"""Benchmark continual learning trainers.

Provides trainers for the benchmark paper with configurable:
- Architecture (Transformer/Mamba/Hybrid via separate job registrations)
- Schedule (WSD, cosine rewarm, WSD+reset)
- Optimization (Full or LoRA via separate job registrations)

Non-LoRA jobs: continual/train/benchmark{,_mamba,_hybrid}
LoRA jobs: continual/train/benchmark{,_mamba,_hybrid}_lora
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, List, Type, TypeVar
from typing import cast as type_cast

import numpy as np


import optax
from loguru import logger

from theseus.config import field, configure
from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.registry import job
from theseus.model.models import GPT, Mamba, Hybrid
from theseus.training.schedules import SCHEDULES
from theseus.model.module import Module
from theseus.experiments.continual.abcd import (
    ABCDBaseTrainer,
    ABCDConfig,
)
from theseus.training.lora import (
    LoRAConfig,
    LoRATrainer,
    LoRATrainerConfig,
)


# ======================================================================
# Config
# ======================================================================


@dataclass
class BenchmarkConfig(ABCDConfig):
    """Config for non-LoRA benchmark runs."""

    schedule_type: str = field("optimization/schedule", default="wsd")
    reset_optimizer_at_boundaries: bool = field(
        "optimization/reset_optimizer", default=False
    )


@dataclass
class BenchmarkLoRAConfig(BenchmarkConfig, LoRATrainerConfig):
    """Config for LoRA benchmark runs.

    Inherits BenchmarkConfig (ABCD multi-stage + schedule/reset) and
    LoRATrainerConfig (pre/post LoRA token budgets + datasets).

    The pre-LoRA phase uses ABCDConfig's ``total_tokens``/``datasets``
    for multi-stage full-param training.  The post-LoRA phase uses
    ``post_lora_tokens``/``post_lora_datasets`` from LoRATrainerConfig
    for adapter-only training.

    LoRA hyperparameters (rank, alpha, target_modules) live under
    ``optimization/lora/`` and are read via ``configure(LoRAConfig)``.
    """

    pass


# ======================================================================
# Non-LoRA Benchmark Trainer
# ======================================================================

BC = TypeVar("BC", bound=BenchmarkConfig)
M = TypeVar("M", bound=Module)


class BenchmarkBaseTrainer(ABCDBaseTrainer[BC, M], Generic[BC, M]):
    """Benchmark trainer with schedule-variant boundary handling.

    Extends ABCDBaseTrainer with:
    - Cosine rewarm: rebuilds schedule at each boundary
    - WSD+Reset: resets optimizer state at each boundary
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"

    def _rebuild_schedule_for_stage(self, stage_idx: int) -> None:
        """Rebuild the LR schedule for the remaining tokens in this stage.

        Used for cosine rewarm: each stage gets a fresh warmup + cosine decay.
        """
        remaining_tokens = sum(self.args.total_tokens[stage_idx:])
        remaining_steps = int(
            remaining_tokens / self.args.batch_size / self.args.block_size
        )
        if remaining_steps <= 0:
            return

        sched_fn, sched_cfg_cls = SCHEDULES["cosine_rewarm"]
        sched_cfg = configure(sched_cfg_cls)
        self.scheduler = sched_fn(remaining_steps, sched_cfg)  # type: ignore[operator]
        self.tx = self._optimizer()
        new_opt_state = self.tx.init(self.state.params)
        self.state = self.state.replace(opt_state=new_opt_state)

    def _reset_optimizer_state(self) -> None:
        """Reset optimizer moment estimates to zero."""
        new_opt_state = self.tx.init(self.state.params)
        self.state = self.state.replace(opt_state=new_opt_state)

    def _on_dataset_boundary(
        self, old_idx: int, new_idx: int, current_ntok: int
    ) -> None:
        """Extends parent boundary handling with schedule-variant logic."""
        super()._on_dataset_boundary(old_idx, new_idx, current_ntok)

        if self.args.schedule_type == "cosine_rewarm":
            self._rebuild_schedule_for_stage(new_idx)
        elif self.args.reset_optimizer_at_boundaries:
            self._reset_optimizer_state()

    # batch() is inherited from ABCDBaseTrainer — no override needed.
    # The _on_dataset_boundary hook handles schedule/reset at boundaries.


# ======================================================================
# LoRA Benchmark Trainer
# ======================================================================

BLC = TypeVar("BLC", bound=BenchmarkLoRAConfig)


class BenchmarkLoRABaseTrainer(BenchmarkBaseTrainer[BLC, M], Generic[BLC, M]):
    """Benchmark trainer with LoRA support.

    Three-level data loading:
    1. Pre-LoRA ABCD stages (full params) — uses ABCDConfig datasets/tokens
    2. LoRA injection at boundary
    3. Post-LoRA ABCD stages (LoRA params only) — uses post_lora_datasets/tokens
    """

    CONFIG = BenchmarkLoRAConfig  # type: ignore[assignment]

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return ABCDBaseTrainer._config() + [LoRAConfig]

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Initialize data for both pre-LoRA and post-LoRA phases."""
        super()._init_data(spec)

        self.lora_config = configure(LoRAConfig)
        self._in_lora_phase = False
        self._pre_lora_total = sum(self.args.total_tokens)

        # Post-LoRA strategies
        from theseus.training.flywheel.strategy import Strategy

        self._post_strategies = [
            Strategy(spec, self.args.block_size, ds)
            for ds in self.args.post_lora_datasets
        ]
        self._post_train_dls = [
            s.get_async_batches(self._train_batch_rows, split="train")
            for s in self._post_strategies
        ]
        self._post_val_dls = [
            s.get_async_batches(self._val_batch_rows, split="val", deterministic_key=32)
            for s in self._post_strategies
        ]

        # Post-LoRA segment boundaries
        self._post_segment_starts: List[int] = [self._pre_lora_total]
        for i in range(len(self.args.post_lora_tokens) - 1):
            self._post_segment_starts.append(
                self._post_segment_starts[-1] + self.args.post_lora_tokens[i]
            )
        self._post_segment_ends: List[int] = [
            self._post_segment_starts[i] + self.args.post_lora_tokens[i]
            for i in range(len(self.args.post_lora_tokens))
        ]

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        topology = super()._init_topology(spec)
        # Extend total steps to include post-LoRA tokens
        self.total_steps = int(
            (sum(self.args.total_tokens) + sum(self.args.post_lora_tokens))
            / self.args.batch_size
            / self.args.block_size
        )
        return topology

    def _transition_to_lora(self) -> None:
        """Freeze base params and inject LoRA adapters.

        Reuses the shared transition logic from lora.py, then resets
        the ABCD dataloader index for the post-LoRA stages.
        """
        from theseus.training.lora import transition_to_lora

        transition_to_lora(self)
        self._current_dl_idx = 0  # reset for post-LoRA stages

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        current_ntok = (
            (self.global_step_counter_ // self.accumulate_steps)
            * self.args.batch_size
            * self.args.block_size
        )

        # Phase transition: pre-LoRA -> LoRA
        if not self._in_lora_phase and current_ntok >= self._pre_lora_total:
            logger.info("LORA | transitioning at {} tokens", current_ntok)
            self.save(Path("pre_lora_checkpoint"))
            self._transition_to_lora()

        if self._in_lora_phase:
            # Post-LoRA: simple stage switching
            dls = self._post_train_dls if slice == "train" else self._post_val_dls
            for i, end in enumerate(self._post_segment_ends):
                if current_ntok < end:
                    stage = i
                    break
            else:
                stage = len(self._post_segment_ends) - 1
            return type_cast(PyTree[np.ndarray], dls[stage].get_batch())
        else:
            # Pre-LoRA: use parent ABCD batch logic
            return super().batch(slice)

    # forward() is inherited from the LoRA code path — during the pre-LoRA
    # phase params are normal model params and merge_lora_params is skipped;
    # during post-LoRA, params = {"lora_A": ..., "lora_B": ...} and the
    # merge happens automatically.  See LoRATrainer.forward.
    forward = LoRATrainer.forward


# ======================================================================
# Registered Jobs: Non-LoRA (3)
# ======================================================================


@job("continual/train/benchmark")
class BenchmarkTransformer(BenchmarkBaseTrainer[BenchmarkConfig, GPT]):
    MODEL = GPT
    CONFIG = BenchmarkConfig


@job("continual/train/benchmark_mamba")
class BenchmarkMamba(BenchmarkBaseTrainer[BenchmarkConfig, Mamba]):
    MODEL = Mamba
    CONFIG = BenchmarkConfig


@job("continual/train/benchmark_hybrid")
class BenchmarkHybrid(BenchmarkBaseTrainer[BenchmarkConfig, Hybrid]):
    MODEL = Hybrid
    CONFIG = BenchmarkConfig


# ======================================================================
# Registered Jobs: LoRA (3)
# ======================================================================


@job("continual/train/benchmark_lora")
class BenchmarkTransformerLoRA(BenchmarkLoRABaseTrainer[BenchmarkLoRAConfig, GPT]):
    MODEL = GPT
    CONFIG = BenchmarkLoRAConfig


@job("continual/train/benchmark_mamba_lora")
class BenchmarkMambaLoRA(BenchmarkLoRABaseTrainer[BenchmarkLoRAConfig, Mamba]):
    MODEL = Mamba
    CONFIG = BenchmarkLoRAConfig


@job("continual/train/benchmark_hybrid_lora")
class BenchmarkHybridLoRA(BenchmarkLoRABaseTrainer[BenchmarkLoRAConfig, Hybrid]):
    MODEL = Hybrid
    CONFIG = BenchmarkLoRAConfig
