"""
KL Divergence Trainer

Two-stage trainer: stage 1 is standard pretraining, stage 2 enforces a
customizable KL penalty against the stage-1 reference policy.

Stage switching follows the same token-counting approach used by the
ABCD continual-learning trainers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast as type_cast
from typing import Any, Dict, Optional, List, Type, Generic, TypeVar

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jax_random
import flax
import flax.linen
from flax.training import train_state
from jax.experimental import multihost_utils

import optax
import wandb
from loguru import logger

from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.flywheel.strategy import Strategy, Sampling, DatasetStyle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KLConfig:
    """KL divergence penalty configuration."""

    beta: float = field("optimization/kl/beta", default=0.1)


@dataclass
class KLDivergenceTrainerConfig(BaseTrainerConfig):
    """Config for two-stage KL-divergence trainer.

    ``total_tokens`` is a two-element list: [stage1_tokens, stage2_tokens].
    ``datasets`` is a two-element list of sampling lists, one per stage.
    """

    total_tokens: List[int] = field(
        "training/tokens",
        default_factory=lambda: [1_000_000_000, 100_000_000],
    )  # type: ignore

    datasets: List[List[Sampling]] = field(  # type: ignore
        "training/dataset",
        default_factory=lambda: [
            [Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD)],
            [Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD)],
        ],
    )


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------


class KLDivergenceTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    """Train state carrying a frozen reference-policy snapshot and KL weight."""

    base: PyTree[Any]  # reference policy params (frozen)
    beta: float  # KL penalty weight (0 = disabled)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

C = TypeVar("C", bound=KLDivergenceTrainerConfig)


class KLDivergenceTrainer(BaseTrainer[C, M], Generic[C, M]):
    """Two-stage trainer with KL-divergence penalty.

    * **Stage 1** – standard language-model pretraining (cross-entropy only).
    * **Stage 2** – pretraining loss *plus* ``beta * KL(policy || reference)``
      where the reference policy is a frozen snapshot taken at the stage
      boundary.

    The KL penalty is approximated as the difference in per-token NLL
    between the current model and the reference model on the same batch:
    ``kl_penalty = policy_loss - sg(reference_loss)``.
    """

    CONFIG = KLDivergenceTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return BaseTrainer._config() + [KLConfig]

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsds"

    # ------------------------------------------------------------------
    # Topology – total tokens are the *sum* of all stages
    # ------------------------------------------------------------------

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        assert spec.topology is not None, (
            "Topology must be provided to perform training"
        )
        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas
        self.total_steps = int(
            sum(self.args.total_tokens)
            / self.args.batch_size
            / self.args.block_size
        )
        return topology

    # ------------------------------------------------------------------
    # State – includes reference-policy slot
    # ------------------------------------------------------------------

    def _init_state(self, params: PyTree[jax.Array]) -> None:
        self.kl_config = configure(KLConfig)

        self.scheduler: optax._src.base.Schedule = self._schedule()
        self.tx = self._optimizer()

        # Stage 0 is pure pretraining – start with beta=0.
        # The configured kl_config.beta is applied at the stage boundary.
        beta = 0.0

        def make_state(p: PyTree[jax.Array]) -> KLDivergenceTrainState:
            # Start with a zero-valued base; it gets populated at the
            # stage-1 → stage-2 boundary.
            base = jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), p
            )
            return type_cast(
                KLDivergenceTrainState,
                KLDivergenceTrainState.create(
                    apply_fn=self.model.apply,
                    params=p,
                    base=base,
                    tx=self.tx,
                    beta=beta,
                ),
            )

        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(
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

    # ------------------------------------------------------------------
    # Data – one Strategy per stage, with precomputed segment boundaries
    # ------------------------------------------------------------------

    def _init_data(self, spec: ExecutionSpec) -> None:
        n = len(self.args.datasets)
        self.strategies = [
            Strategy(spec, self.args.block_size, ds) for ds in self.args.datasets
        ]

        self._train_batch_rows = (
            self.per_device_batch_size * self.local_replicas * self.accumulate_steps
        )
        self.train_dls = [
            s.get_async_batches(self._train_batch_rows, split="train")
            for s in self.strategies
        ]

        self._val_batch_rows = max(
            self.per_device_batch_size * self.local_replicas,
            (
                self.args.validation_steps
                // (self.per_device_batch_size * self.local_replicas)
            )
            * (self.per_device_batch_size * self.local_replicas),
        )
        self.val_dls = [
            s.get_async_batches(
                self._val_batch_rows, split="val", deterministic_key=32
            )
            for s in self.strategies
        ]

        # Track which stage we are in
        self._current_stage: int = 0

        # Precompute token boundaries
        self._segment_starts: List[int] = [0]
        for i in range(n - 1):
            self._segment_starts.append(
                self._segment_starts[-1] + self.args.total_tokens[i]
            )
        self._segment_ends: List[int] = [
            self._segment_starts[i] + self.args.total_tokens[i] for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Batch – hard-switch between stages, snapshot reference at boundary
    # ------------------------------------------------------------------

    def _current_token_position(self) -> int:
        return (
            (self.global_step_counter_ // self.accumulate_steps)
            * self.args.batch_size
            * self.args.block_size
        )

    def _stage_for_token(self, ntok: int) -> int:
        """Return the stage index for a given token position."""
        for i, end in enumerate(self._segment_ends):
            if ntok < end:
                return i
        return len(self._segment_ends) - 1

    def _snapshot_reference(self) -> None:
        """Snapshot current params into state.base as the reference policy.

        The cast operates element-wise on each host's local shards –
        no cross-host gather is performed and sharding is preserved.
        Barriers ensure every host completes the snapshot before any
        host proceeds to use the new reference.
        """
        multihost_utils.sync_global_devices("kl_snapshot:start")
        # .astype on sharded arrays preserves sharding; each host only
        # touches its local shards, so no full-param materialisation.
        new_base = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), self.state.params
        )
        self.state = self.state.replace(base=new_base)
        multihost_utils.sync_global_devices("kl_snapshot:end")
        if self.main_process():
            logger.info("KL | reference policy snapshot taken")

    def _on_stage_boundary(self, old_stage: int, new_stage: int) -> None:
        """Called when transitioning between stages.

        By default, snapshots the reference policy when entering stage 1
        (i.e. the second stage, index 1).  Subclasses may override for
        more complex behaviour.
        """
        # Evaluate at boundary
        multihost_utils.sync_global_devices("eval_barrier:start")
        self.inference.state = self.state
        eval_metrics = self.inference.evaluate()
        multihost_utils.sync_global_devices("eval_barrier:end")

        if self.main_process():
            logger.info("EVAL | {}", eval_metrics)
            wandb.log(
                eval_metrics,
                step=(self.global_step_counter_ // self.accumulate_steps),
            )

        logger.info(
            "STAGE | switching from stage {} to stage {} at {} tokens",
            old_stage,
            new_stage,
            self._current_token_position(),
        )
        self.save(Path(f"boundary_{old_stage}_{new_stage}"))

        if self.main_process():
            wandb.log(
                {
                    "stage/index": new_stage,
                    "stage/switch_at_tokens": self._current_token_position(),
                },
                step=self.global_step_counter_ // self.accumulate_steps,
            )

        # Snapshot reference policy and activate KL penalty when entering
        # stage 1 (the KL stage).
        if new_stage >= 1:
            self._snapshot_reference()
            self.state = self.state.replace(beta=self.kl_config.beta)
            if self.main_process():
                logger.info("KL | beta set to {}", self.kl_config.beta)

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        from typing import cast as type_cast

        current_ntok = self._current_token_position()
        new_stage = self._stage_for_token(current_ntok)

        if new_stage != self._current_stage:
            self._on_stage_boundary(self._current_stage, new_stage)
            self._current_stage = new_stage

        dls = self.train_dls if slice == "train" else self.val_dls
        return type_cast(PyTree[np.ndarray], dls[self._current_stage].get_batch())

    # ------------------------------------------------------------------
    # Forward – standard CE + KL penalty (stage >= 1 only)
    # ------------------------------------------------------------------

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        intermediates: bool = False,
    ) -> Any:
        kl_state = type_cast(KLDivergenceTrainState, state)
        batch_dict = type_cast(Dict[str, jax.Array], batch)

        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)
        rngs = {"dropout": dropout_key} if dropout_key is not None else {}

        # Policy forward pass
        if intermediates:
            (logits, policy_loss), mutated = kl_state.apply_fn(
                {"params": params},
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
            logits, policy_loss = kl_state.apply_fn(
                {"params": params},
                x,
                y,
                padding_mask=padding_mask,
                deterministic=deterministic,
                rngs=rngs,
            )
            meta = {}

        # KL penalty: beta * (policy_loss - sg(reference_loss))
        # The reference forward always executes (beta is a traced value),
        # but in stage 0 beta=0 so the penalty is zeroed out numerically.
        beta = kl_state.beta

        ref_loss = jax.lax.stop_gradient(
            kl_state.apply_fn(
                {"params": kl_state.base},
                x,
                y,
                padding_mask=padding_mask,
                deterministic=True,
            )[1]
        )

        kl_penalty = policy_loss - ref_loss
        total_loss = policy_loss + beta * kl_penalty

        meta.update(
            {
                "kl/policy_loss": jax.lax.stop_gradient(policy_loss),
                "kl/ref_loss": ref_loss,
                "kl/penalty": jax.lax.stop_gradient(kl_penalty),
                "kl/beta": beta,
                "kl/total_loss": jax.lax.stop_gradient(total_loss),
            }
        )

        return logits, total_loss, meta
