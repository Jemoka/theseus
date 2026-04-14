"""Benchmark continual learning trainers.

Provides trainers for the benchmark paper with configurable:
- Architecture (Transformer/Mamba/Hybrid via separate job registrations)
- Schedule (WSD, cosine rewarm, WSD+reset)
- Optimization (Full or LoRA via separate job registrations)

Non-LoRA jobs: continual/train/benchmark{,_mamba,_hybrid}
LoRA jobs: continual/train/benchmark{,_mamba,_hybrid}_lora
"""

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, Generic, List, Type, TypeVar
from typing import cast as type_cast

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

import optax
import wandb
from loguru import logger

from theseus.config import field, configure
from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.registry import job
from theseus.model.models import GPT, Mamba, Hybrid
from theseus.training.schedules import SCHEDULES
from theseus.training.flywheel.strategy import Sampling, DatasetStyle
from theseus.model.module import Module
from theseus.experiments.continual.abcd import (
    ABCDBaseTrainer,
    ABCDConfig,
    FadeConfig,
)
from theseus.training.lora import (
    LoRAConfig,
    LoRATrainState,
    param_filter,
    inject_lora_params,
    merge_lora_params,
    count_lora_params,
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
class BenchmarkLoRAConfig(ABCDConfig):
    """Config for LoRA benchmark runs.

    Two levels of ABCD stages: pre-LoRA (full params) and post-LoRA
    (adapter-only). Each side has its own multi-stage dataset schedule.
    """

    schedule_type: str = field("optimization/schedule", default="wsd")
    reset_optimizer_at_boundaries: bool = field(
        "optimization/reset_optimizer", default=False
    )

    lora: LoRAConfig = dataclass_field(default_factory=LoRAConfig)

    # Pre-LoRA datasets/tokens are the main ABCDConfig ones (total_tokens, datasets)
    # Post-LoRA datasets/tokens
    post_lora_tokens: List[int] = field(
        "training/post_lora_tokens",
        default_factory=lambda: [100_000_000],
    )
    post_lora_datasets: List[List[Sampling]] = field(
        "training/post_lora_dataset",
        default_factory=lambda: [
            [Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD)],
        ],
    )
    post_lora_fade: FadeConfig = dataclass_field(default_factory=FadeConfig)


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
        # Reset optimizer state for the new schedule
        new_opt_state = self.tx.init(self.state.params)
        self.state = self.state.replace(opt_state=new_opt_state)

    def _reset_optimizer_state(self) -> None:
        """Reset optimizer moment estimates to zero."""
        new_opt_state = self.tx.init(self.state.params)
        self.state = self.state.replace(opt_state=new_opt_state)

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        """Batch with schedule-variant boundary handling."""
        from typing import cast as type_cast

        current_ntok = (
            (self.global_step_counter_ // self.accumulate_steps)
            * self.args.batch_size
            * self.args.block_size
        )

        weights = self._compute_fade_weights(current_ntok)
        primary_idx = max(range(len(weights)), key=lambda i: weights[i])

        # Boundary detection
        if primary_idx != self._current_dl_idx:
            # Run parent boundary logic (eval, save, log)
            if self._current_dl_idx == 0 and self.args.skip_first_dataset_validation:
                self.args.validation_interval = self._real_validation_interval

            multihost_utils.sync_global_devices("eval_barrier:start")
            self.inference.state = self.state
            eval_metrics = self.inference.evaluate()
            multihost_utils.sync_global_devices("eval_barrier:end")

            if self.main_process():
                logger.info("EVAL | {}", eval_metrics)
                step = self.global_step_counter_ // self.accumulate_steps
                wandb.log(eval_metrics, step=step)

            logger.info(
                "DATASET | switching primary from {} to {} at {} tokens",
                self._current_dl_idx,
                primary_idx,
                current_ntok,
            )
            self.save(Path(f"boundary_{self._current_dl_idx}_{primary_idx}"))

            if self.main_process():
                wandb.log(
                    {
                        "dataset/index": primary_idx,
                        "dataset/switch_at_tokens": current_ntok,
                    },
                    step=self.global_step_counter_ // self.accumulate_steps,
                )

            # Schedule-variant handling
            if self.args.schedule_type == "cosine_rewarm":
                self._rebuild_schedule_for_stage(primary_idx)
            elif self.args.reset_optimizer_at_boundaries:
                self._reset_optimizer_state()

            self._current_dl_idx = primary_idx

        # Draw batches from active dataloaders (same as parent)
        dls = self.train_dls if slice == "train" else self.val_dls
        total_rows = (
            self._train_batch_rows if slice == "train" else self._val_batch_rows
        )
        counts = self._distribute_batch(weights, total_rows)

        batch_parts: List[dict[str, Any]] = []
        for i, count in enumerate(counts):
            if count > 0:
                batch_data = dls[i].get_batch()
                batch_parts.append({k: v[:count] for k, v in batch_data.items()})

        combined: dict[str, Any] = {}
        for key in batch_parts[0]:
            combined[key] = np.concatenate([p[key] for p in batch_parts], axis=0)

        n = combined[next(iter(combined))].shape[0]
        perm = np.random.permutation(n)
        combined = {k: v[perm] for k, v in combined.items()}

        return type_cast(PyTree[np.ndarray], combined)


# ======================================================================
# LoRA Benchmark Trainer
# ======================================================================

BLC = TypeVar("BLC", bound=BenchmarkLoRAConfig)


class BenchmarkLoRABaseTrainer(BenchmarkBaseTrainer[BLC, M], Generic[BLC, M]):  # type: ignore[type-var]
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
        """Freeze base params and inject LoRA adapters."""
        multihost_utils.sync_global_devices("lora_transition:start")

        base_params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), self.state.params
        )
        mask = param_filter(self.state.params, self.lora_config.target_modules)
        lora_A, lora_B = inject_lora_params(
            self.state.params, mask, self.lora_config.rank, jax.random.PRNGKey(0)
        )

        count = count_lora_params(lora_A, lora_B)
        if self.main_process():
            logger.info("LORA | injected {} trainable params", count)

        lora_tx = optax.adam(learning_rate=self.scheduler)

        def make_lora_state(base: Any, a: Any, b: Any) -> LoRATrainState:
            return type_cast(
                LoRATrainState,
                LoRATrainState.create(  # type: ignore
                    apply_fn=self.model.apply,
                    params=self.state.params,
                    base_params=base,
                    lora_A=a,
                    lora_B=b,
                    tx=lora_tx,
                    lora_alpha=self.lora_config.alpha,
                    lora_rank=self.lora_config.rank,
                ),
            )

        self.state = make_lora_state(base_params, lora_A, lora_B)
        self._in_lora_phase = True
        self._current_dl_idx = 0  # reset for post-LoRA stages

        multihost_utils.sync_global_devices("lora_transition:end")

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

    @staticmethod
    def forward(
        state: Any,
        params: Any,
        batch: Any,
        key: Any = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        batch_dict = type_cast(Dict[str, jax.Array], batch)
        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax.random.split(key)
        rngs = {"dropout": dropout_key} if dropout_key is not None else {}

        effective_params = params
        if hasattr(state, "lora_A") and hasattr(state, "base_params"):
            lora_state = type_cast(LoRATrainState, state)
            effective_params = merge_lora_params(
                lora_state.base_params,
                lora_state.lora_A,
                lora_state.lora_B,
                lora_state.lora_alpha,
                lora_state.lora_rank,
            )

        if intermediates:
            (logits, loss), mutated = state.apply_fn(
                {"params": effective_params},
                x,
                y,
                padding_mask=padding_mask,
                deterministic=deterministic,
                rngs=rngs,
                mutable=["intermediates", "plots"],
            )
            meta: Dict[str, Any] = {
                "intermediates": mutated.get("intermediates", {}),
                "plots": mutated.get("plots", {}),
            }
        else:
            logits, loss = state.apply_fn(
                {"params": effective_params},
                x,
                y,
                padding_mask=padding_mask,
                deterministic=deterministic,
                rngs=rngs,
            )
            meta = {}

        return logits, loss, meta


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
