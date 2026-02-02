"""
a very basic trainer
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Type, Dict, Any, Optional, List

import jax
import jax.numpy as jnp
from jax import random as jax_random

import flax
from flax.training import train_state

import optax

import wandb
from loguru import logger

from theseus.base import ExecutionSpec
from theseus.job import CheckpointedJob
from theseus.config import field, current_config, configure

from theseus.model.module import Module

from theseus.training.optimizers import OPTIMIZERS
from theseus.training.schedules import SCHEDULES
from theseus.training.utils import find_accumulation_steps

M = TypeVar("M", bound=Module)


@dataclass
class BaseTrainerConfig:
    # Training hyperparameters
    batch_size: int = field("training/batch_size", default=512)
    per_device_batch_size: int = field(
        "training/per_device_batch_size", default=8
    )  # TODO! this should be automatically detected somehow
    total_steps: int = field(
        "training/total_steps"
    )  # TODO: this should be computed based on the dataset spec

    # Learning rate schedule (WSD: Warmup-Stable-Decay)
    lr: float = field("optimization/lr", default=3e-4)
    warmup_pct: float = field("training/warmup_pct", default=0.01)
    decay_pct: float = field("training/decay_pct", default=0.1)

    # Some Architecture
    # We need to know block size for data handling; the model will
    # ask for the rest of the architecture parameters itself
    block_size: int = field("architecture/block_size", default=512)

    # Logging/checkpointing
    report_interval: int = field("logging/report_interval", default=32)
    checkpoint_interval: int = field("training/checkpoint_interval", default=1024)
    validation_interval: int = field("training/validation_interval", default=512)
    validation_steps: int = field("training/validation_steps", default=2048)

    # W&B
    wandb: bool = field("logging/wandb", default=False)


class BaseTrainer(CheckpointedJob[BaseTrainerConfig], Generic[M]):
    """
    Generic pretrainer for GPT-style models.
    """

    MODEL: Type[M]

    @classmethod
    def config(cls) -> List[Type[Any]]:
        cfg = [BaseTrainerConfig, *cls.MODEL.gather()]

        if isinstance(cls.optimizer(), str):
            _, optim_cfg = OPTIMIZERS.get(cls.optimizer(), (None, None))
            if optim_cfg is not None:
                cfg.append(optim_cfg)

        return cfg

    def _schedule(self) -> optax.base.Schedule:
        """build learning rate schedule from config"""
        if self.schedule() is None:
            return optax.constant_schedule(self.args.lr)

        sched_name = self.schedule()
        if isinstance(sched_name, optax.base.Schedule):
            return sched_name

        sched, cfg = SCHEDULES[sched_name]  # type: ignore

        return sched(self.args.total_steps, configure(cfg))

    def _optimizer(self) -> optax.GradientTransformation:
        optim_name = self.optimizer()
        if isinstance(optim_name, optax.GradientTransformation):
            return optim_name

        optim, cfg = OPTIMIZERS[optim_name]

        return optim(self._schedule(), configure(cfg))

    @classmethod
    def optimizer(cls) -> str | optax.GradientTransformation:
        """return either an optimizer from the optimizer library, or a custom optax optimizer"""

        return "adamw"

    @classmethod
    def schedule(cls) -> Optional[str | optax.base.Schedule]:
        """return either a learning rate schedule, a schedule name from the library, or nothing to use a constant lr"""

        return None

    def __init__(self, spec: ExecutionSpec) -> None:
        """Build a basic trainer

        Args:
            spec (ExecutionSpec): execution specification
            abstract (bool, optional): whether not we will load in a checkpoint later
                                       and thus should not initialize parameters now.

        Raises:
            AssertionError: if topology is not provided in spec
        """

        super().__init__(spec)

        # first get the requested topology from spec
        assert spec.topology is not None, (
            "Topology must be provided to perform training"
        )

        topology = spec.topology
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas

        # compute our batch size
        self.per_device_batch_size, self.accumulate_steps = find_accumulation_steps(
            self.args.batch_size, self.args.per_device_batch_size, topology
        )

        # Total micro-batches to process per node
        self.total_batches = self.args.total_steps * self.accumulate_steps

        # Log a bunch of things
        if self.main_process():
            logger.info(
                "BATCHING | {} batchsize/node * ({} local * {} prox = {} dp) * {} accumulation = {} batchsize",
                self.per_device_batch_size,
                self.local_replicas,
                jax.process_count(),
                self.replicas,
                self.accumulate_steps,
                self.args.batch_size,
            )
            logger.info(
                "STEPS | {} micro batches // {} accumulation = {} steps",
                self.total_batches,
                self.accumulate_steps,
                self.total_batches // self.accumulate_steps,
            )
            logger.info(
                "TOKENS | {} steps * {} batchsize * {} blocksize = {} tokens",
                self.total_batches // self.accumulate_steps,
                self.args.batch_size,
                self.args.block_size,
                (self.total_batches // self.accumulate_steps)
                * self.args.batch_size
                * self.args.block_size,
            )

            assert current_config() is not None, (
                "cannot locate configuration in context!"
            )
            cfg: Dict[Any, Any] = dict(current_config())  # type: ignore

            wandb.init(
                project=spec.project or "theseus",
                config=cfg,
                mode=None if self.args.wandb else "disabled",
                name=self.spec.name,
                group=self.spec.group,
                resume="allow",
                id=spec.id,
            )

        # initialize model from the config in thin air
        self.model: M = configure(self.MODEL)

        # Initialize random keys
        self.key, init_key, self.dropout_key = jax_random.split(self.key, num=3)

        # Initialize model
        dummy_input = jnp.ones((1, self.args.block_size), dtype=jnp.int32)
        variables = self.model.init(init_key, dummy_input)
        params = variables["params"]

        # build the optimizer
        self.tx = self._optimizer()
        logger.info(f"OPTIMIZER | {self.tx}")

        # build state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.tx
        )  # type: ignore
        self.total_params = (
            sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        )

        if self.main_process():
            logger.info(f"MODEL | Total Parameters: {self.total_params:.2f}m")

        # Shard the state
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
            flax.linen.get_partition_spec(self.state),
            self.mesh,
            rules=tuple(self.model.sharding),
        )
        self.state = jax.device_put(self.state, self.state_sharding)

        # Initialize counters
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf")

        # weeeeeeeeeeee
        # print the model
        if self.main_process():
            logger.info(self.model)
