"""Benchmark continual learning trainers.

Provides trainers for the benchmark paper with configurable:
- Architecture (Transformer/Mamba/Hybrid/MoE via separate job registrations)
- Schedule (WSD, cosine rewarm, WSD+reset)
- Optimization (Full or LoRA via separate job registrations)

Non-LoRA jobs: continual/train/benchmark{,_mamba,_hybrid,_moe}
LoRA jobs: continual/train/benchmark{,_mamba,_hybrid,_moe}_lora
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar
from typing import cast as type_cast

import numpy as np
import wandb


import optax
from jax.experimental import multihost_utils
from loguru import logger

from theseus.config import field, configure
from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.registry import job
from theseus.model.models import GPT, Mamba, Hybrid, MoEGPT
from theseus.model.module import Module
from theseus.experiments.continual.abcd import (
    ABCDBaseTrainer,
    ABCDConfig,
    _make_eval_bar_chart,
    _make_eval_timeline_chart,
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

    def _reset_optimizer_state(self) -> None:
        """Reset optimizer moment estimates to zero."""
        new_opt_state = self.tx.init(self.state.params)
        self.state = self.state.replace(opt_state=new_opt_state)

    def _on_dataset_boundary(
        self, old_idx: int, new_idx: int, current_ntok: int
    ) -> None:
        """Extends parent boundary handling with WSD+reset logic.

        Cosine rewarm doesn't need special boundary handling here — the
        schedule is built upfront with all boundary positions baked in
        via ``CosineRewarmConfig.stage_tokens``.
        """
        super()._on_dataset_boundary(old_idx, new_idx, current_ntok)

        if self.args.reset_optimizer_at_boundaries:
            self._reset_optimizer_state()


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

        # LoRA-phase plotting state: kept separate from ABCD's pre-LoRA
        # _eval_history / _boundary_tokens so the two timelines don't mix.
        self._lora_eval_history: Dict[str, List[Tuple[int, float]]] = {}
        self._lora_boundary_tokens: List[int] = []
        self._current_post_stage: int = 0

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

    def _on_lora_boundary(self, label: str, current_ntok: int) -> None:
        """Run eval + emit LoRA-specific plots + save checkpoint at a LoRA boundary.

        Mirrors ``ABCDBaseTrainer._on_dataset_boundary`` but keeps its
        own eval history / timeline (``eval/lora_timeline``) so the
        LoRA-phase adaptation signal is plotted separately from the
        pre-LoRA pretraining timeline.
        """
        multihost_utils.sync_global_devices("lora_eval_barrier:start")
        self.inference.state = self.state
        eval_metrics = self.inference.evaluate()
        multihost_utils.sync_global_devices("lora_eval_barrier:end")

        if self.main_process():
            logger.info("LORA EVAL | {}", eval_metrics)
            step = self.global_step_counter_ // self.accumulate_steps
            wandb.log(eval_metrics, step=step)
            self._lora_boundary_tokens.append(current_ntok)

            if len(eval_metrics) > 0:
                boundary_label = f"lora_{label}"
                metrics_snapshot = dict(eval_metrics)
                self.plotter.plot(
                    lambda m=metrics_snapshot,  # type: ignore[misc]
                    lbl=boundary_label: _make_eval_bar_chart(m, lbl),
                    step=step,
                )

                for k, v in metrics_snapshot.items():
                    self._lora_eval_history.setdefault(k, []).append(
                        (current_ntok, float(v))
                    )
                history_snap = {k: list(v) for k, v in self._lora_eval_history.items()}
                boundaries_snap = list(self._lora_boundary_tokens)
                self.plotter.plot(
                    lambda h=history_snap,  # type: ignore[misc]
                    b=boundaries_snap: _make_eval_timeline_chart(
                        h, b, timeline_key="eval/lora_timeline"
                    ),
                    step=step,
                )

        logger.info("LORA BOUNDARY | {} at {} tokens", label, current_ntok)
        self.save(Path(f"lora_boundary_{label}"))

        if self.main_process():
            wandb.log(
                {
                    "lora/boundary_label": label,
                    "lora/switch_at_tokens": current_ntok,
                },
                step=self.global_step_counter_ // self.accumulate_steps,
            )

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
            self._current_post_stage = 0
            self._on_lora_boundary("pre_to_post", current_ntok)

        if self._in_lora_phase:
            # Post-LoRA: simple stage switching
            dls = self._post_train_dls if slice == "train" else self._post_val_dls
            for i, end in enumerate(self._post_segment_ends):
                if current_ntok < end:
                    stage = i
                    break
            else:
                stage = len(self._post_segment_ends) - 1

            if stage != self._current_post_stage:
                self._on_lora_boundary(
                    f"post_{self._current_post_stage}_to_{stage}", current_ntok
                )
                self._current_post_stage = stage

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


@job("continual/train/benchmark_moe")
class BenchmarkMoE(BenchmarkBaseTrainer[BenchmarkConfig, MoEGPT]):
    MODEL = MoEGPT
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


@job("continual/train/benchmark_moe_lora")
class BenchmarkMoELoRA(BenchmarkLoRABaseTrainer[BenchmarkLoRAConfig, MoEGPT]):
    MODEL = MoEGPT
    CONFIG = BenchmarkLoRAConfig
