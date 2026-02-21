import optax
import wandb
import numpy as np
from pathlib import Path

from typing import List, Tuple, Generic, TypeVar
from loguru import logger
from jax.experimental import multihost_utils

from dataclasses import dataclass
from theseus.config import field
from theseus.model.models import GPT
from theseus.base import Topology, ExecutionSpec
from theseus.training.trainer import BaseTrainer, BaseTrainerConfig, M
from theseus.training.huggingface import HFTrainerConfig
from theseus.training.flywheel.strategy import Sampling, DatasetStyle, Strategy
from theseus.evaluation.base import Evaluator
from theseus.experiments.models.gpt import EvaluateGPT


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


C = TypeVar("C", bound=ABCDConfig)


@dataclass
class ABCDHFConfig(ABCDConfig, HFTrainerConfig): ...


class ABCDBaseTrainer(BaseTrainer[C, M], Generic[C, M]):
    """Standard continual learning: sequential shift and plasticity."""

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsds"  # we want to use a contsant LR

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
        """Initialize dataset strategy and data loaders."""
        # make dataset strategy

        self.strategies = [
            Strategy(spec, self.args.block_size, i) for i in self.args.datasets
        ]
        self.train_dls = [
            i.get_async_batches(
                self.per_device_batch_size
                * self.local_replicas
                * self.accumulate_steps,
                split="train",
            )
            for i in self.strategies
        ]
        self.val_dls = [
            i.get_async_batches(
                (
                    (
                        self.args.validation_steps
                        // (self.per_device_batch_size * self.local_replicas)
                    )
                    * (self.per_device_batch_size * self.local_replicas)
                ),
                split="val",
                deterministic_key=32,
            )
            for i in self.strategies
        ]

        # Track current dataset index for logging switches
        self._current_dl_idx: int = 0

    def batch(self, slice: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get the next batch from the dataset strategy"""

        # based on global_step_counter, get the right dl
        current_ntok = (
            (self.global_step_counter_ // self.accumulate_steps)
            * self.args.batch_size
            * self.args.block_size
        )

        # find which dataloader to use based on cumulative token thresholds
        cumulative = 0
        dl_idx = len(self.args.total_tokens) - 1
        for i, tokens in enumerate(self.args.total_tokens):
            cumulative += tokens
            if current_ntok < cumulative:
                dl_idx = i
                break

        # Log dataset switch if the index changed
        if dl_idx != self._current_dl_idx:
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
                "DATASET | switching from dataset {} to {} at {} tokens",
                self._current_dl_idx,
                dl_idx,
                current_ntok,
            )
            self.save(Path(f"boundary_{self._current_dl_idx}_{dl_idx}"))

            if self.main_process():
                wandb.log(
                    {
                        "dataset/index": dl_idx,
                        "dataset/switch_at_tokens": current_ntok,
                    },
                    step=self.global_step_counter_ // self.accumulate_steps,
                )
            self._current_dl_idx = dl_idx

        x: np.ndarray
        y: np.ndarray
        padding_mask: np.ndarray

        if slice == "train":
            x, y, padding_mask = self.train_dls[dl_idx].get_batch()
        else:
            x, y, padding_mask = self.val_dls[dl_idx].get_batch()

        return x, y, padding_mask


class ABCDTrainer(ABCDBaseTrainer[ABCDConfig, GPT]):
    MODEL = GPT
    CONFIG = ABCDConfig

    def evaluator(self) -> Evaluator[GPT]:
        return EvaluateGPT.from_trainer(self)
