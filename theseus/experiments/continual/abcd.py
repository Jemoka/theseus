import optax
import numpy as np

from typing import List, Tuple

from theseus.config import field
from theseus.model.models import GPT
from theseus.base import Topology, ExecutionSpec
from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.training.flywheel.strategy import Sampling, DatasetStyle, Strategy


class ABCDConfig(BaseTrainerConfig):
    total_tokens: List[int] = field(
        "training/tokens",
        default_factory=lambda: [1_000_000_000, 10_000, 10_000, 10_000, 10_000],
    )  # type: ignore
    datasets: List[List[Sampling]] = field(  # type: ignore
        "training/dataset",
        default_factory=lambda: [
            [
                Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD),
            ],
            [
                Sampling(name="fineweb", rate=0.5, style=DatasetStyle.PMD),
                Sampling(name="mnli", rate=0.5, style=DatasetStyle.PMD),
            ],
            [
                Sampling(name="fineweb", rate=0.5, style=DatasetStyle.PMD),
                Sampling(name="qqp", rate=0.5, style=DatasetStyle.PMD),
            ],
            [
                Sampling(name="fineweb", rate=0.5, style=DatasetStyle.PMD),
                Sampling(name="sst2", rate=0.5, style=DatasetStyle.PMD),
            ],
            [
                Sampling(name="fineweb", rate=0.5, style=DatasetStyle.PMD),
                Sampling(name="siqa", rate=0.5, style=DatasetStyle.PMD),
            ],
        ],
    )


class ABCDTrainer(BaseTrainer[GPT]):
    MODEL = GPT

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return None  # we want to use a contsant LR

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
        self.total_steps = (
            sum(self.args.total_tokens) // self.args.batch_size // self.args.block_size  # type: ignore
        )
        return topology

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Initialize dataset strategy and data loaders."""
        # make dataset strategy

        self.strategies = [
            Strategy(spec, self.args.block_size, i)  # type: ignore
            for i in self.args.datasets
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
        dl_idx = len(self.args.total_tokens) - 1  # type: ignore
        for i, tokens in enumerate(self.args.total_tokens):  # type: ignore
            cumulative += tokens
            if current_ntok < cumulative:
                dl_idx = i
                break

        x: np.ndarray
        y: np.ndarray
        padding_mask: np.ndarray

        if slice == "train":
            x, y, padding_mask = self.train_dls[dl_idx].get_batch()
        else:
            x, y, padding_mask = self.val_dls[dl_idx].get_batch()

        return x, y, padding_mask
