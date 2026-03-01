import optax
import wandb
import numpy as np
from pathlib import Path

from typing import Any, Dict, List, Generic, TypeVar, Callable
from loguru import logger
from jax.experimental import multihost_utils

from dataclasses import dataclass, field as dataclass_field
from theseus.config import field
from theseus.model.models import GPT
from theseus.base import Topology, ExecutionSpec, PyTree
from theseus.training.base import BaseTrainer, BaseTrainerConfig, M
from theseus.training.huggingface import HFTrainerConfig
from theseus.training.flywheel.strategy import Sampling, DatasetStyle, Strategy
from theseus.plot import PALETTE


@dataclass
class FadeConfig:
    """Configuration for gradual dataset fade transitions.

    Controls how datasets blend during boundary transitions instead of
    hard-switching.

    Parameters:
        overlap: Size of the blending region as a fraction of the smaller
            adjacent segment. 0 = hard switch (original behavior),
            1 = maximum overlap (fade spans the entire smaller segment).
        curve: Shape of the blending function. One of:
            - "linear": constant-rate crossfade
            - "cosine": smooth S-curve (slow start/end, fast middle)
            - "sigmoid": steep S-curve controlled by ``steepness``
        steepness: Slope of the sigmoid curve at the midpoint.
            Only used when ``curve="sigmoid"``. Higher values produce
            a sharper transition. Default 10.0.
        per_boundary_overlap: Per-boundary overlap overrides.  When
            non-empty, must have length ``len(datasets) - 1``.  Each
            entry replaces ``overlap`` for the corresponding boundary.
    """

    overlap: float = field("training/fade/overlap", default=0.0)
    curve: str = field("training/fade/curve", default="linear")
    steepness: float = field("training/fade/steepness", default=10.0)
    per_boundary_overlap: List[float] = field(
        "training/fade/per_boundary_overlap",
        default_factory=list,
    )


@dataclass
class ABCDConfig(BaseTrainerConfig):
    total_tokens: List[int] = field(
        "training/tokens",
        default_factory=lambda: [
            1_000_000_000,
            100_000_000,
            100_000_000,
            100_000_000,
            100_000_000,
        ],
    )  # type: ignore

    warmup_pct: float = field("optimization/warmup_pct", default=0.01)
    decay_pct: float = field("optimization/decay_pct", default=0.01)
    constant_pct: float = field("optimization/constant_pct", default=0.30)

    datasets: List[List[Sampling]] = field(  # type: ignore
        "training/dataset",
        default_factory=lambda: [
            [
                Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD),
            ],
            [
                Sampling(name="mnli", rate=1, style=DatasetStyle.PADDED),
            ],
            [
                Sampling(name="qqp", rate=1, style=DatasetStyle.PADDED),
            ],
            [
                Sampling(name="sst2", rate=1, style=DatasetStyle.PADDED),
            ],
            [
                Sampling(name="siqa", rate=1, style=DatasetStyle.PADDED),
            ],
        ],
    )
    evaluations: List[str] = field(
        "eval/evaluations",
        default_factory=lambda: [
            "mnli",
            "qqp",
            "sst2",
            "siqa",
        ],
    )

    fade: FadeConfig = dataclass_field(default_factory=FadeConfig)


C = TypeVar("C", bound=ABCDConfig)


@dataclass
class ABCDHFConfig(ABCDConfig, HFTrainerConfig): ...


def _make_eval_bar_chart(
    eval_metrics: dict[str, float], boundary_label: str
) -> Dict[str, Any]:
    """Create a bar chart of evaluation results at a dataset boundary.

    Called on the Plotter worker thread where matplotlib is already
    initialized and apply_theme() has been applied.
    """
    from matplotlib import pyplot as plt

    names = list(eval_metrics.keys())
    scores = list(eval_metrics.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 3.5))
    bars = ax.bar(names, scores, color=colors, width=0.6)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation at boundary {boundary_label}")
    ax.set_ylim(0, max(max(scores) * 1.15, 0.1) if scores else 1.0)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return {f"eval/boundary_{boundary_label}": fig}


class ABCDBaseTrainer(BaseTrainer[C, M], Generic[C, M]):
    """Standard continual learning with optional gradual fade between datasets.

    When ``fade.overlap > 0``, adjacent dataset segments overlap so that
    the model transitions gradually rather than hard-switching at
    boundaries.  All strategies are created and their async batch workers
    started at init time; during the fade region batches are drawn from
    both the outgoing and incoming dataloaders and combined proportionally.
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsds"  # we want to use a contsant LR

    # ------------------------------------------------------------------
    # Fade helpers
    # ------------------------------------------------------------------

    def _get_fade_curve(self) -> Callable[[float], float]:
        """Return the blending function selected by config."""
        fade = self.args.fade
        if fade.curve == "linear":
            return lambda t: t
        elif fade.curve == "cosine":
            return lambda t: float((1 - np.cos(t * np.pi)) / 2)
        elif fade.curve == "sigmoid":
            k = fade.steepness
            return lambda t: float(1.0 / (1.0 + np.exp(-k * (t - 0.5))))
        raise ValueError(
            f"Unknown fade curve '{fade.curve}'. Choose from: linear, cosine, sigmoid"
        )

    def _compute_fade_weights(self, current_ntok: int) -> List[float]:
        """Compute per-dataset sampling weights for the current token position.

        Each dataset segment has a core region where its weight is 1 and
        optional fade-in / fade-out ramps at its boundaries with
        neighbouring segments.  The ramp length is
        ``overlap * min(T_left, T_right)`` centred on the boundary.

        Returns a normalised weight vector (sums to 1).
        """
        n = len(self.args.total_tokens)
        fade = self.args.fade

        # Per-boundary overlap values
        if fade.per_boundary_overlap and len(fade.per_boundary_overlap) == n - 1:
            overlaps = list(fade.per_boundary_overlap)
        else:
            overlaps = [fade.overlap] * max(n - 1, 0)

        curve_fn = self._get_fade_curve()

        weights: List[float] = []
        for i in range(n):
            seg_start = self._segment_starts[i]
            seg_end = self._segment_ends[i]

            # Fade-in region (boundary with previous segment)
            if i > 0:
                ov = overlaps[i - 1]
                fade_half = (
                    ov
                    * min(
                        self.args.total_tokens[i - 1],
                        self.args.total_tokens[i],
                    )
                    / 2
                )
                fade_in_start = seg_start - fade_half
                fade_in_end = seg_start + fade_half
            else:
                fade_in_start = float(seg_start)
                fade_in_end = float(seg_start)

            # Fade-out region (boundary with next segment)
            if i < n - 1:
                ov = overlaps[i]
                fade_half = (
                    ov
                    * min(
                        self.args.total_tokens[i],
                        self.args.total_tokens[i + 1],
                    )
                    / 2
                )
                fade_out_start = seg_end - fade_half
                fade_out_end = seg_end + fade_half
            else:
                fade_out_start = float(seg_end)
                fade_out_end = float(seg_end)

            # Outside this segment's active region?
            if current_ntok < fade_in_start or current_ntok >= fade_out_end:
                weights.append(0.0)
                continue

            w = 1.0
            # Fade-in ramp
            if fade_in_end > fade_in_start and current_ntok < fade_in_end:
                t = (current_ntok - fade_in_start) / (fade_in_end - fade_in_start)
                w = min(w, curve_fn(t))

            # Fade-out ramp
            if fade_out_end > fade_out_start and current_ntok >= fade_out_start:
                t = (current_ntok - fade_out_start) / (fade_out_end - fade_out_start)
                w = min(w, 1.0 - curve_fn(t))

            weights.append(max(0.0, w))

        # Normalise
        total = sum(weights)
        if total > 0:
            return [w / total for w in weights]

        # Fallback: hard assignment to the segment that owns this position
        cumulative = 0
        for i, tokens in enumerate(self.args.total_tokens):
            cumulative += tokens
            if current_ntok < cumulative:
                weights[i] = 1.0
                return weights
        weights[-1] = 1.0
        return weights

    @staticmethod
    def _distribute_batch(weights: List[float], total: int) -> List[int]:
        """Distribute *total* batch rows across dataloaders proportional to *weights*.

        Uses largest-remainder allocation so the counts always sum to *total*.
        """
        counts = [0] * len(weights)
        active = [(i, w) for i, w in enumerate(weights) if w > 0]
        if not active:
            return counts

        for i, w in active:
            counts[i] = int(w * total)

        remainder = total - sum(counts)
        fracs = sorted(
            [(w * total - int(w * total), i) for i, w in active],
            reverse=True,
        )
        for _, i in fracs[:remainder]:
            counts[i] += 1

        return counts

    # ------------------------------------------------------------------
    # Trainer lifecycle overrides
    # ------------------------------------------------------------------

    def _init_topology(self, spec: ExecutionSpec) -> Topology:
        """Initialize topology, mesh, and compute total steps."""
        # first get the requested topology from spec
        assert spec.topology is not None, (
            "Topology must be provided to perform training"
        )

        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas
        self.total_steps = int(
            sum(self.args.total_tokens) / self.args.batch_size / self.args.block_size
        )
        return topology

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Initialize dataset strategies and data loaders.

        Creates one Strategy per dataset grouping and starts all async
        batch workers immediately so they are warm when the fade schedule
        begins sampling from them.
        """
        # Validate per-boundary overlap length if provided
        n = len(self.args.datasets)
        fade = self.args.fade
        if fade.per_boundary_overlap and len(fade.per_boundary_overlap) != n - 1:
            raise ValueError(
                f"per_boundary_overlap must have {n - 1} entries "
                f"(one per boundary), got {len(fade.per_boundary_overlap)}"
            )

        # Create all strategies up front
        self.strategies = [
            Strategy(spec, self.args.block_size, i) for i in self.args.datasets
        ]

        # Start async batch workers for every strategy immediately
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
                self._val_batch_rows,
                split="val",
                deterministic_key=32,
            )
            for s in self.strategies
        ]

        # Track current primary dataset index for boundary logging
        self._current_dl_idx: int = 0

        # Precompute segment boundaries (token positions)
        self._segment_starts: List[int] = [0]
        for i in range(n - 1):
            self._segment_starts.append(
                self._segment_starts[-1] + self.args.total_tokens[i]
            )
        self._segment_ends: List[int] = [
            self._segment_starts[i] + self.args.total_tokens[i] for i in range(n)
        ]

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        """Return the next training or validation batch.

        When ``fade.overlap == 0`` this behaves identically to the
        original hard-switch logic.  With ``overlap > 0`` it draws rows
        from multiple dataloaders in proportion to their fade weights and
        shuffles them together.
        """
        from typing import cast as type_cast

        # Current token position
        current_ntok = (
            (self.global_step_counter_ // self.accumulate_steps)
            * self.args.batch_size
            * self.args.block_size
        )

        # Compute per-dataset weights
        weights = self._compute_fade_weights(current_ntok)
        primary_idx = max(range(len(weights)), key=lambda i: weights[i])

        # ---------- Boundary evaluation when primary dataset changes ----------
        if primary_idx != self._current_dl_idx:
            multihost_utils.sync_global_devices("eval_barrier:start")
            self.inference.state = self.state
            eval_metrics = self.inference.evaluate()
            multihost_utils.sync_global_devices("eval_barrier:end")

            if self.main_process():
                logger.info("EVAL | {}", eval_metrics)
                step = self.global_step_counter_ // self.accumulate_steps
                wandb.log(eval_metrics, step=step)

                if self.plotter is not None and len(eval_metrics) > 0:
                    boundary_label = f"{self._current_dl_idx}_to_{primary_idx}"
                    metrics_snapshot = dict(eval_metrics)
                    self.plotter.plot(
                        lambda m=metrics_snapshot, l=boundary_label: _make_eval_bar_chart(m, l),
                        step=step,
                    )

            logger.info(
                "DATASET | switching primary from {} to {} at {} tokens (weights: {})",
                self._current_dl_idx,
                primary_idx,
                current_ntok,
                [f"{w:.3f}" for w in weights],
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
            self._current_dl_idx = primary_idx

        # ---------- Draw and merge batches from active dataloaders ----------
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

        # Concatenate all parts
        combined: dict[str, Any] = {}
        for key in batch_parts[0]:
            combined[key] = np.concatenate([p[key] for p in batch_parts], axis=0)

        # Shuffle so micro-batches see a uniform mix
        n = combined[next(iter(combined))].shape[0]
        perm = np.random.permutation(n)
        combined = {k: v[perm] for k, v in combined.items()}

        return type_cast(PyTree[np.ndarray], combined)


class ABCDTrainer(ABCDBaseTrainer[ABCDConfig, GPT]):
    MODEL = GPT
    CONFIG = ABCDConfig
