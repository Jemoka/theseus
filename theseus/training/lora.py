"""LoRA (Low-Rank Adaptation) Trainer.

Two-phase trainer: phase 1 trains full parameters on pre-LoRA datasets,
then freezes and injects low-rank adapters, phase 2 trains only LoRA
parameters on post-LoRA datasets.

Since this extends RestoreableJob, users can restore from a checkpoint
and set pre_lora_datasets/tokens to empty to skip straight to LoRA
fine-tuning from a pretrained checkpoint.
"""

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import cast as type_cast
from typing import Any, Dict, List, Optional, Tuple, Type, Generic, TypeVar

import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen
from flax.training import train_state
from jax.experimental import multihost_utils

import optax
from loguru import logger

from theseus.base import PyTree, Topology, ExecutionSpec
from theseus.config import field, configure
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.flywheel.strategy import Strategy, Sampling, DatasetStyle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Low-rank adaptation parameters."""

    rank: int = field("optimization/lora/rank", default=16)
    alpha: float = field("optimization/lora/alpha", default=16.0)
    target_modules: List[str] = field(
        "optimization/lora/target_modules",
        default_factory=lambda: ["kernel"],
    )


@dataclass
class LoRATrainerConfig(BaseTrainerConfig):
    """Config for two-phase LoRA trainer.

    Pre-LoRA phase trains full parameters, then LoRA adapters are injected
    and only they are trained in the post-LoRA phase.
    """

    lora: LoRAConfig = dataclass_field(default_factory=LoRAConfig)

    pre_lora_tokens: List[int] = field(
        "training/pre_lora_tokens",
        default_factory=lambda: [1_000_000_000],
    )
    pre_lora_datasets: List[List[Sampling]] = field(
        "training/pre_lora_dataset",
        default_factory=lambda: [
            [Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD)],
        ],
    )

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


# ---------------------------------------------------------------------------
# LoRA parameter utilities
# ---------------------------------------------------------------------------


def param_filter(
    params: PyTree[jax.Array],
    target_modules: List[str],
) -> PyTree[bool]:
    """Create a boolean mask over the param tree.

    Returns True for leaves whose path contains any of ``target_modules``.
    This is the filter that determines which parameters get LoRA adapters.
    """

    def _match(path: str) -> bool:
        return any(t in path for t in target_modules)

    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    mask_flat: list[bool] = []
    for keypath, _ in flat:
        path_str = "/".join(str(k) for k in keypath)
        mask_flat.append(_match(path_str))
    return treedef.unflatten(mask_flat)  # type: ignore[no-any-return, attr-defined]


def inject_lora_params(
    params: PyTree[jax.Array],
    mask: PyTree[bool],
    rank: int,
    key: jax.Array,
) -> Tuple[PyTree[Optional[jax.Array]], PyTree[Optional[jax.Array]]]:
    """Create LoRA A/B matrices for each targeted parameter.

    For a parameter of shape (in_features, out_features):
    - A: (in_features, rank) — initialized from normal(0, 1/rank)
    - B: (rank, out_features) — initialized to zeros

    Returns two pytrees (lora_A, lora_B) with same structure as params.
    Non-targeted leaves are zero arrays (same shape as base param) so
    the tree structure is compatible with jax.tree_util operations.
    """
    flat_params, treedef = jax.tree_util.tree_flatten(params)
    flat_mask = jax.tree_util.tree_leaves(mask)

    a_flat: list[Any] = []
    b_flat: list[Any] = []
    key_idx = 0
    for param, is_target in zip(flat_params, flat_mask):
        if is_target and param.ndim == 2:
            in_f, out_f = param.shape
            k1 = jax.random.fold_in(key, key_idx)
            A = jax.random.normal(k1, (in_f, rank), dtype=param.dtype) / rank
            B = jnp.zeros((rank, out_f), dtype=param.dtype)
            a_flat.append(A)
            b_flat.append(B)
        else:
            # Non-targeted: use None sentinel
            a_flat.append(None)
            b_flat.append(None)
        key_idx += 1

    return treedef.unflatten(a_flat), treedef.unflatten(b_flat)  # type: ignore[attr-defined]


def merge_lora_params(
    base_params: PyTree[jax.Array],
    lora_A: PyTree[Any],
    lora_B: PyTree[Any],
    alpha: float,
    rank: int,
) -> PyTree[jax.Array]:
    """Merge LoRA into base: W_eff = W + (alpha/rank) * A @ B."""
    scale = alpha / rank

    def _merge(base: jax.Array, a: Any, b: Any) -> jax.Array:
        if a is None or b is None:
            return base
        delta = scale * (a @ b)
        return base + delta.astype(base.dtype)  # type: ignore[no-any-return]

    return jax.tree_util.tree_map(  # type: ignore[no-any-return]
        _merge,
        base_params,
        lora_A,
        lora_B,
        is_leaf=lambda x: x is None,
    )


def count_lora_params(lora_A: PyTree[Any], lora_B: PyTree[Any]) -> int:
    """Count total trainable LoRA parameters."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(lora_A):
        if leaf is not None and hasattr(leaf, "size"):
            total += leaf.size
    for leaf in jax.tree_util.tree_leaves(lora_B):
        if leaf is not None and hasattr(leaf, "size"):
            total += leaf.size
    return total


# ---------------------------------------------------------------------------
# Custom train state
# ---------------------------------------------------------------------------


class LoRATrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    """Train state for LoRA phase: frozen base + trainable adapters."""

    base_params: PyTree[Any]  # frozen base parameters
    lora_A: PyTree[Any]  # trainable LoRA A matrices
    lora_B: PyTree[Any]  # trainable LoRA B matrices
    lora_alpha: float
    lora_rank: int


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

C = TypeVar("C", bound=LoRATrainerConfig)


class LoRATrainer(BaseTrainer[C, M], Generic[C, M]):
    """Two-phase LoRA trainer.

    Phase 1 (pre-LoRA): Full-parameter training on ``pre_lora_datasets``.
    Phase 2 (post-LoRA): Freeze base, inject LoRA adapters, train only
    adapters on ``post_lora_datasets``.

    The transition happens automatically at the token boundary between
    pre-LoRA and post-LoRA phases.
    """

    CONFIG = LoRATrainerConfig  # type: ignore[assignment]

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        return BaseTrainer._config() + [LoRAConfig]

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        assert spec.topology is not None
        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas
        self.total_steps = int(
            (sum(self.args.pre_lora_tokens) + sum(self.args.post_lora_tokens))
            / self.args.batch_size
            / self.args.block_size
        )
        return topology

    # ------------------------------------------------------------------
    # State – starts as standard TrainState; switches to LoRATrainState
    # ------------------------------------------------------------------

    def _init_state(self, params: PyTree[jax.Array]) -> None:
        """Initialize standard train state for the pre-LoRA phase."""
        self.lora_config = configure(LoRAConfig)
        self.scheduler = self._schedule()
        self.tx = self._optimizer()

        def make_state(p: PyTree[jax.Array]) -> train_state.TrainState:
            return train_state.TrainState.create(  # type: ignore
                apply_fn=self.model.apply, params=p, tx=self.tx
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

    def _transition_to_lora(self) -> None:
        """Freeze base params and inject LoRA adapters.

        Rebuilds the optimizer to only train LoRA parameters.
        """
        multihost_utils.sync_global_devices("lora_transition:start")

        base_params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), self.state.params
        )
        mask = param_filter(self.state.params, self.lora_config.target_modules)
        lora_A, lora_B = inject_lora_params(
            self.state.params, mask, self.lora_config.rank, jax.random.PRNGKey(0)
        )

        lora_param_count = count_lora_params(lora_A, lora_B)
        if self.main_process():
            logger.info(
                "LORA | injected {} trainable params (rank={}, alpha={})",
                lora_param_count,
                self.lora_config.rank,
                self.lora_config.alpha,
            )

        # Build optimizer for LoRA params only
        lora_tx = optax.adam(learning_rate=self.scheduler)

        # Create LoRA train state
        def make_lora_state(
            base: PyTree[jax.Array],
            a: PyTree[Any],
            b: PyTree[Any],
        ) -> LoRATrainState:
            return type_cast(
                LoRATrainState,
                LoRATrainState.create(  # type: ignore
                    apply_fn=self.model.apply,
                    params=self.state.params,  # current full params
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

        multihost_utils.sync_global_devices("lora_transition:end")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _init_data(self, spec: ExecutionSpec) -> None:
        # Pre-LoRA strategies
        self._pre_strategies = [
            Strategy(spec, self.args.block_size, ds)
            for ds in self.args.pre_lora_datasets
        ]
        # Post-LoRA strategies
        self._post_strategies = [
            Strategy(spec, self.args.block_size, ds)
            for ds in self.args.post_lora_datasets
        ]

        self._train_batch_rows = (
            self.per_device_batch_size * self.local_replicas * self.accumulate_steps
        )

        self._pre_train_dls = [
            s.get_async_batches(self._train_batch_rows, split="train")
            for s in self._pre_strategies
        ]
        self._post_train_dls = [
            s.get_async_batches(self._train_batch_rows, split="train")
            for s in self._post_strategies
        ]

        self._val_batch_rows = max(
            self.per_device_batch_size * self.local_replicas,
            (
                self.args.validation_steps
                // (self.per_device_batch_size * self.local_replicas)
            )
            * (self.per_device_batch_size * self.local_replicas),
        )
        self._pre_val_dls = [
            s.get_async_batches(self._val_batch_rows, split="val", deterministic_key=32)
            for s in self._pre_strategies
        ]
        self._post_val_dls = [
            s.get_async_batches(self._val_batch_rows, split="val", deterministic_key=32)
            for s in self._post_strategies
        ]

        # State tracking
        self._in_lora_phase = False
        self._current_stage = 0
        self._pre_lora_total = sum(self.args.pre_lora_tokens)

        # Segment boundaries for pre and post phases
        self._pre_segment_ends = []
        cumulative = 0
        for t in self.args.pre_lora_tokens:
            cumulative += t
            self._pre_segment_ends.append(cumulative)

        self._post_segment_ends = []
        cumulative = self._pre_lora_total
        for t in self.args.post_lora_tokens:
            cumulative += t
            self._post_segment_ends.append(cumulative)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def _current_token_position(self) -> int:
        return (
            (self.global_step_counter_ // self.accumulate_steps)
            * self.args.batch_size
            * self.args.block_size
        )

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        from typing import cast as type_cast

        current_ntok = self._current_token_position()

        # Check for phase transition
        if not self._in_lora_phase and current_ntok >= self._pre_lora_total:
            logger.info("LORA | transitioning to LoRA phase at {} tokens", current_ntok)
            self.save(Path("pre_lora_checkpoint"))
            self._transition_to_lora()
            self._current_stage = 0

        # Determine which dataloader to use
        if self._in_lora_phase:
            dls = self._post_train_dls if slice == "train" else self._post_val_dls
            # Find current post-lora stage
            for i, end in enumerate(self._post_segment_ends):
                if current_ntok < end:
                    stage = i
                    break
            else:
                stage = len(self._post_segment_ends) - 1
        else:
            dls = self._pre_train_dls if slice == "train" else self._pre_val_dls
            # Find current pre-lora stage
            for i, end in enumerate(self._pre_segment_ends):
                if current_ntok < end:
                    stage = i
                    break
            else:
                stage = len(self._pre_segment_ends) - 1

        return type_cast(PyTree[np.ndarray], dls[stage].get_batch())

    # ------------------------------------------------------------------
    # Forward — merges LoRA params before forward pass in LoRA phase
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
        batch_dict = type_cast(Dict[str, jax.Array], batch)
        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax.random.split(key)
        rngs = {"dropout": dropout_key} if dropout_key is not None else {}

        # In LoRA phase, merge base + LoRA params for forward
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
