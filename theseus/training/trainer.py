"""
a very basic trainer
"""

from pathlib import Path
from dataclasses import asdict, dataclass, is_dataclass
from pprint import pformat
from typing import Generic, TypeVar, Type, Dict, Any, Optional, List, Tuple, Callable

from omegaconf import OmegaConf

import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jax_random
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

import flax
from flax.training import train_state

import optax

import wandb
from loguru import logger

from theseus.base import ExecutionSpec
from theseus.job import RestoreableJob
from theseus.config import field, current_config, configure

from theseus.model.module import Module

from theseus.base import PyTree, Axis, Topology
from theseus.training.optimizers import OPTIMIZERS
from theseus.training.schedules import SCHEDULES
from theseus.training.utils import (
    find_accumulation_steps,
)
from theseus.training.flywheel.strategy import Strategy, Sampling, DatasetStyle
from theseus.evaluation.base import Evaluator, EvaluatorConfig
from theseus.data.tokenizer import TokenizerConfig

M = TypeVar("M", bound=Module)


@dataclass
class BaseTrainerConfig:
    # Training hyperparameters
    batch_size: int = field("training/batch_size", default=512)
    per_device_batch_size: int = field(
        "training/per_device_batch_size", default=-1
    )  # -1 = auto-estimate based on VRAM
    total_tokens: int = field("training/tokens", default=1_000_000_000)

    # Learning rate schedule (WSD: Warmup-Stable-Decay)
    lr: float = field("optimization/lr", default=3e-4)
    warmup_pct: float = field("training/warmup_pct", default=0.01)
    decay_pct: float = field("training/decay_pct", default=0.1)

    # run validation or not?
    validate: bool = field("training/validation", default=True)
    evaluate: bool = field("training/evaluate", default=True)

    # dataset
    datasets: List[Sampling] = field(
        "training/dataset",
        default_factory=lambda: [
            Sampling(name="fineweb", rate=1, style=DatasetStyle.PMD)
        ],
    )

    # evaluation
    evaluations: List[str] = field(
        "eval/evaluations",
        default_factory=lambda: [],
    )

    # Some Architecture
    # We need to know block size for data handling; the model will
    # ask for the rest of the architecture parameters itself
    block_size: int = field("architecture/block_size", default=512)

    # Logging/checkpointing
    report_interval: int = field("logging/report_interval", default=32)
    checkpoint_interval: int = field("logging/checkpoint_interval", default=1024)
    validation_interval: int = field("logging/validation_interval", default=512)
    validation_steps: int = field("training/validation_steps", default=2048)

    # W&B
    wandb: bool = field("logging/wandb", default=False)


C = TypeVar("C", bound=BaseTrainerConfig)


class BaseTrainer(RestoreableJob[C], Generic[C, M]):
    """
    Generic pretrainer for GPT-style models.
    """

    MODEL: Type[M]
    CONFIG: Type[C]

    @classmethod
    def config(cls) -> List[Type[Any]]:
        return cls._config() + [cls.CONFIG]

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        cfg: List[Type[Any]] = [*cls.MODEL.gather(), EvaluatorConfig, TokenizerConfig]

        if isinstance(cls.optimizer(), str):
            _, optim_cfg = OPTIMIZERS.get(cls.optimizer(), (None, None))
            if optim_cfg is not None:
                cfg.append(optim_cfg)

        if isinstance(cls.schedule(), str):
            _, sched_cfg = SCHEDULES.get(cls.schedule(), (None, None))  # type: ignore
            if sched_cfg is not None:
                cfg.append(sched_cfg)

        return cfg

    def _schedule(self) -> optax._src.base.Schedule:
        """build learning rate schedule from config"""
        if self.schedule() is None:
            return optax.constant_schedule(self.args.lr)

        sched_name = self.schedule()
        if not isinstance(sched_name, str):
            return sched_name

        sched, cfg = SCHEDULES[sched_name]

        return sched(self.total_steps, configure(cfg))  # type: ignore

    def _optimizer(self) -> optax.GradientTransformation:
        optim_name = self.optimizer()
        if isinstance(optim_name, optax.GradientTransformation):
            return optim_name

        optim, cfg = OPTIMIZERS[optim_name]

        return optim(self.scheduler, configure(cfg))

    def __init__(self, spec: ExecutionSpec) -> None:
        """Build a basic trainer

        Args:
            spec (ExecutionSpec): execution specification

        Raises:
            AssertionError: if topology is not provided in spec
        """

        super().__init__(spec)
        logger.info(f"TOPOLOGY | \n{spec.model_dump_json(indent=2)}\n")

        self.args = configure(self.CONFIG)
        config_dump = asdict(self.args) if is_dataclass(self.args) else self.args
        logger.info(f"CONFIG | \n{pformat(config_dump, sort_dicts=False)}\n")
        topology = self._init_topology(spec)
        params = self._init_model()
        self._init_state(params)
        self._init_batch_config(topology)
        self._init_wandb(spec)
        self._init_data(spec)
        self._init_counters_and_eval()

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
            self.args.total_tokens / self.args.batch_size / self.args.block_size
        )
        return topology

    @staticmethod
    def sharded_init(
        model: Any, key: jax.Array, *args: Any, mesh: jax.sharding.Mesh, **kwargs: Any
    ) -> Tuple[Any, Any]:
        """Initialize model params sharded across mesh from the start.

        Uses eval_shape to compute sharding spec without materializing params,
        then JITs init with out_shardings so arrays land on correct devices directly.

        Returns (variables, var_sharding).
        """
        shapes = jax.eval_shape(model.init, key, *args, **kwargs)
        var_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
            flax.linen.get_partition_spec(shapes),
            mesh,
            rules=tuple(model.sharding),
        )
        variables = jax.jit(model.init, out_shardings=var_sharding)(
            key, *args, **kwargs
        )
        return variables, var_sharding

    def _init_model(self) -> PyTree[jax.Array]:
        """Initialize model and random keys, return initial sharded params."""
        # initialize model from the config in thin air
        self.model: M = configure(self.MODEL)

        # Initialize random keys
        self.key, init_key, self.dropout_key = jax_random.split(self.key, num=3)

        # Initialize model with params sharded across mesh from the start
        dummy_input = jnp.ones((1, self.args.block_size), dtype=jnp.int32)
        variables, _ = self.sharded_init(
            self.model, init_key, dummy_input, mesh=self.mesh
        )
        return variables["params"]  # type: ignore[no-any-return]

    def _init_state(self, params: PyTree[jax.Array]) -> None:
        """Build optimizer, scheduler, and sharded train state."""
        # build the optimizer
        self.scheduler: optax._src.base.Schedule = self._schedule()
        self.tx = self._optimizer()

        # build state sharded from the start — use eval_shape to get state pytree
        # structure (different from param pytree: includes opt_state, step, etc.)
        # then JIT create with out_shardings so optimizer buffers land on correct devices
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

    def _init_batch_config(self, topology: Topology) -> None:
        """Compute batch size, accumulation steps, and log configuration."""
        # Compute batch size (auto-estimate or manual override)
        if self.args.per_device_batch_size < 0:
            raise ValueError(
                "You specified -1 for the per-device batch size, but auto-estimation requires running with dispatch harness which you didn't appear to use; please either use `theseus bootstrap` or `theseus submit` endpoints OR specify a positive integer for `per_device_batch_size` in your config."
            )
        else:
            fitted_bs = self.args.per_device_batch_size

        self.per_device_batch_size, self.accumulate_steps = find_accumulation_steps(
            self.args.batch_size, fitted_bs, topology
        )

        # Total micro-batches to process per node
        self.total_batches = self.total_steps * self.accumulate_steps

        # Log batch configuration
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

    def _init_wandb(self, spec: ExecutionSpec) -> None:
        """Initialize Weights & Biases logging."""
        if self.main_process():
            assert current_config() is not None, (
                "cannot locate configuration in context!"
            )
            cfg: Dict[Any, Any] = OmegaConf.to_object(current_config())

            wandb.init(
                project=spec.project or "theseus",
                config=cfg,
                mode=None if self.args.wandb else "disabled",
                name=self.spec.name,
                group=self.spec.group,
                resume="allow",
                id=spec.id,
            )
            if self.args.wandb:
                self.spec.id = wandb.run.id
                logger.info(
                    f"WANDB | project={wandb.run.project} run_id={wandb.run.id} "
                    f"run_name={wandb.run.name} url={wandb.run.get_url()}"
                )

    def _init_data(self, spec: ExecutionSpec) -> None:
        """Initialize dataset strategy and data loaders."""
        # make dataset strategy
        self.strategy = Strategy(spec, self.args.block_size, self.args.datasets)
        self.train_dl = self.strategy.get_async_batches(
            self.per_device_batch_size * self.local_replicas * self.accumulate_steps,
            split="train",
        )
        val_batch_size = max(
            self.per_device_batch_size * self.local_replicas,
            (
                self.args.validation_steps
                // (self.per_device_batch_size * self.local_replicas)
            )
            * (self.per_device_batch_size * self.local_replicas),
        )
        self.val_dl = self.strategy.get_async_batches(
            val_batch_size,
            split="val",
            deterministic_key=32,
        )

    def _init_counters_and_eval(self) -> None:
        """Initialize step counters, score tracking, and evaluator."""
        # Initialize counters
        self.global_step_counter_ = 0
        self.best_val_score_ = float("-inf")

        # bake evaluator
        self.inference: Evaluator[M] = self.evaluator()  # type: ignore

        # weeeeeeeeeeee
        # print the model
        if self.main_process():
            logger.info(self.model)

    def evaluator(self) -> Optional[Evaluator[M]]:
        """define what evaluator to use"""
        return None

    @classmethod
    def optimizer(cls) -> str | optax.GradientTransformation:
        """return either an optimizer from the optimizer library, or a custom optax optimizer"""

        return "adamw"

    @classmethod
    def schedule(cls) -> Optional[str | optax._src.base.Schedule]:
        """return either a learning rate schedule, a schedule name from the library, or nothing to use a constant lr"""

        return None

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        """get the next batch from the dataset strategy"""
        from typing import cast as type_cast

        if slice == "train":
            return type_cast(PyTree[np.ndarray], self.train_dl.get_batch())
        else:
            return type_cast(PyTree[np.ndarray], self.val_dl.get_batch())

    def _reshape_batch(self, batch: PyTree[np.ndarray]) -> PyTree[np.ndarray]:
        """Reshape batch for sharding; assumes dict batches."""
        from typing import cast as type_cast

        per = self.per_device_batch_size * self.local_replicas

        def _reshape(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(-1, per, arr.shape[-1])

        return type_cast(PyTree[np.ndarray], jax.tree_util.tree_map(_reshape, batch))

    def _to_global(self, batch: PyTree[np.ndarray]) -> PyTree[jax.Array]:
        """Move host-local numpy batch to global arrays with standard sharding."""
        from typing import cast as type_cast

        pspec = P(None, Axis.BATCH, None)  # type: ignore

        def convert(arr: np.ndarray) -> jax.Array:
            result: jax.Array = multihost_utils.host_local_array_to_global_array(
                arr,
                self.mesh,
                pspec,
            )
            return result

        return type_cast(PyTree[jax.Array], jax.tree_util.tree_map(convert, batch))

    @staticmethod
    def forward(
        state: train_state.TrainState,
        params: PyTree[jax.Array],
        batch: PyTree[jax.Array],
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
        mutable: Optional[list[str]] = None,
        extra_variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # batch is expected to be a dict with x, y, padding_mask keys
        from typing import cast as type_cast

        batch_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], batch)
        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        dropout_key = None
        if not deterministic and key is not None:
            _, dropout_key = jax_random.split(key)

        variables: Dict[str, Any] = {"params": params}
        if extra_variables is not None:
            variables.update(extra_variables)

        kwargs: Dict[str, Any] = {
            "padding_mask": padding_mask,
            "deterministic": deterministic,
        }
        if dropout_key is not None:
            kwargs["rngs"] = {"dropout": dropout_key}

        if mutable is not None:
            (logits, loss), mutated = state.apply_fn(
                variables, x, y, mutable=mutable, **kwargs
            )
            return (logits, loss, {}), mutated
        else:
            logits, loss = state.apply_fn(variables, x, y, **kwargs)
            return logits, loss, {}

    @classmethod
    def train_step(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],  # (S, B, T) each
        key: jax.Array,
        accumulate_steps: int,
    ) -> Tuple[train_state.TrainState, jax.Array, Any]:
        """Compute gradients over S micro-batches and apply one optimizer step.

        Args:
            state: Current training state
            batch: (x, y, padding_mask) each with shape (S, B, T)
                   S = accumulation steps, B = batch size, T = sequence length
            key: PRNG key for dropout
            accumulate_steps: Number of micro-batches (S)

        Returns:
            (updated_state, loss, meta) where meta is the last micro-batch's metadata
        """

        def train_eval(
            state: train_state.TrainState,
            batch: PyTree[jax.Array],  # (B, T) each
            key: jax.Array,
            accumulate_steps: int,
        ) -> Tuple[jax.Array, PyTree[jax.Array], Any]:
            payload = batch

            def loss_fn(params: PyTree[jax.Array]) -> Tuple[jax.Array, Any]:
                logits, loss, meta = cls.forward(
                    state,
                    params,
                    payload,
                    key=key,
                    deterministic=False,
                )
                return loss / accumulate_steps, meta

            (loss, meta), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )  # loss: scalar, grads: PyTree

            return loss, grads, meta

        def reduce(
            carry: Tuple[PyTree[jax.Array], jax.Array, jax.Array],
            batch_item: Any,  # PyTree with single batch (B, T)
        ) -> Tuple[Tuple[PyTree[jax.Array], jax.Array, jax.Array], Any]:
            grad, loss, key = carry
            key, subkey = jax_random.split(key)
            loss_single, grad_single, meta = train_eval(
                state, batch_item, subkey, accumulate_steps
            )

            grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad, grad_single)
            loss_acc = loss + loss_single

            return (grad_acc, loss_acc, key), meta

        grad_zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)

        # scan over S micro-batches, accumulating gradients; collect per-step meta
        loss_sum: jax.Array
        (grad_sum, loss_sum, _), metas = jax.lax.scan(
            reduce, (grad_zero, jnp.array(0.0), key), batch
        )

        # take the last micro-batch's metadata
        last_meta: Any = jax.tree_util.tree_map(lambda x: x[-1], metas)

        state: PyTree[jax.Array] = state.apply_gradients(grads=grad_sum)  # type: ignore

        return state, loss_sum, last_meta

    @classmethod
    def val_step(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],  # (S, B, T) each
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute validation loss over S micro-batches.

        Args:
            state: Current training state
            batch: (x, y, padding_mask) each with shape (S, B, T)
                   S = accumulation size, B = batch size, T = sequence length

        Returns:
            (loss_sum, token_count) for computing mean: loss = loss_sum / token_count
        """

        def reduce(
            carry: Tuple[jax.Array, jax.Array],
            xb_item: Any,  # PyTree with single batch item (B, T)
        ) -> Tuple[Tuple[jax.Array, jax.Array], None]:
            from typing import cast as type_cast

            loss_sum, count = carry

            # Cast to PyTree for forward call
            xb_pytree: PyTree[jax.Array] = type_cast(PyTree[jax.Array], xb_item)
            params_pytree: PyTree[jax.Array] = type_cast(
                PyTree[jax.Array], state.params
            )
            _, loss_i, _ = cls.forward(
                state, params_pytree, xb_pytree, deterministic=True
            )  # loss_i: scalar

            # Extract mask from batch (expected to be dict)
            xb_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], xb_item)
            mask = xb_dict.get("padding_mask")
            if mask is None and "padding_mask_pos" in xb_dict:
                mask = jnp.stack(
                    [xb_dict["padding_mask_pos"], xb_dict["padding_mask_neg"]], axis=1
                )
            assert mask is not None, "No padding mask found in batch"

            n = mask.sum()  # count real tokens: scalar
            return (loss_sum + loss_i * n, count + n), None

        # scan over S micro-batches, accumulating weighted loss
        (loss_sum, count), _ = jax.lax.scan(
            reduce, (jnp.array(0.0), jnp.array(0)), batch
        )

        return loss_sum, count  # scalar arrays

    def __make_valid_step(
        self,
    ) -> Callable[[train_state.TrainState], Tuple[float, Dict[str, float]]]:
        batch = self._to_global(self._reshape_batch(self.batch("val")))
        data_shard = NamedSharding(self.mesh, P(None, Axis.BATCH, None))  # type: ignore

        valid_step_inner_jit = jax.jit(
            self.val_step,
            in_shardings=(self.state_sharding, data_shard),
            out_shardings=(None, None),
        )

        def valid_step_wrapper(
            state: train_state.TrainState,
        ) -> Tuple[float, Dict[str, float]]:
            loss_sum, count = valid_step_inner_jit(state, batch)
            loss_sum = jax.device_get(loss_sum)
            count = jax.device_get(count)

            # if these come back as per-device arrays, reduce them here
            loss_sum_local = float(jnp.sum(loss_sum))
            count_local = float(jnp.sum(count))

            loss_sum_g = multihost_utils.process_allgather(jnp.asarray(loss_sum_local))
            count_g = multihost_utils.process_allgather(jnp.asarray(count_local))

            loss = float(jnp.sum(loss_sum_g) / jnp.sum(count_g))

            score = 1 / loss
            metrics = {"val/loss": loss, "val/score": score}

            return score, metrics

        return valid_step_wrapper

    def __make_train_step(
        self,
    ) -> Callable[
        [
            train_state.TrainState,
            PyTree[jax.Array],
            jax.Array,
            int,
        ],
        Tuple[train_state.TrainState, jax.Array, Any],
    ]:
        data_shard = NamedSharding(self.mesh, P(None, Axis.BATCH, None))  # type: ignore
        train_step = jax.jit(
            self.train_step,
            in_shardings=(self.state_sharding, data_shard, None, None),
            out_shardings=(self.state_sharding, None, None),
            donate_argnums=(0,),
        )
        return train_step

    def train(self) -> None:
        if self.main_process():
            logger.info("BEGIN TRAINING")

        train_step = self.__make_train_step()
        if self.args.validate:
            valid_step = self.__make_valid_step()

        # because sometimes the load function may skip some epochs
        for indx in range(
            self.global_step_counter_, self.total_batches + 1, self.accumulate_steps
        ):
            logger.debug("DATA | {} | START", indx)
            batch = self._to_global(self._reshape_batch(self.batch()))
            logger.debug("DATA | {} | PLACED", indx)

            self.dropout_key, subkey = jax_random.split(self.dropout_key)
            self.state, loss, train_meta = train_step(
                self.state,
                batch,
                subkey,
                self.accumulate_steps,
            )
            logger.debug("COMPUTATION | {} | FINISHED", indx)
            train_metrics = {}

            # perform logging, and then increment
            if (
                (indx % self.accumulate_steps == 0)
                and (indx // self.accumulate_steps) % self.args.report_interval == 0
                and indx != 0
            ):
                multihost_utils.sync_global_devices("report:pre")
                train_metrics["train/lr"] = float(self.scheduler(self.state.step))
                loss_val = float(loss)

                if self.main_process():
                    train_metrics["train/tokens"] = (
                        ((indx + 1) // self.accumulate_steps)
                        * self.args.batch_size
                        * self.args.block_size
                    )
                    train_metrics["train/loss"] = loss_val
                    train_metrics.update(jax.device_get(train_meta))

                    wandb.log(
                        train_metrics,
                        step=indx // self.accumulate_steps,
                    )
                    logger.info(
                        "TRAIN | {}/{} | loss {}",
                        indx // self.accumulate_steps,
                        self.total_batches // self.accumulate_steps,
                        loss_val,
                    )
                multihost_utils.sync_global_devices("report:post")

            if indx % self.accumulate_steps == 0:
                self.global_step_counter_ += self.accumulate_steps

            if self.main_process():
                logger.debug("STEP | {} | {}", indx, train_metrics)

            # save a checkpoint, if needed
            if (
                indx != 0
                and indx % self.accumulate_steps == 0
                and (indx // self.accumulate_steps) % self.args.checkpoint_interval
                == (
                    self.args.checkpoint_interval // 2
                )  # offset checkpoint to not crash with val
            ):
                self.save(
                    Path("ntoks")
                    / str(
                        (
                            ((indx + 1) // self.accumulate_steps)
                            * self.args.batch_size
                            * self.args.block_size
                        )
                    )
                )  # save as number of tokens

            # perform validation and save a checkpoint, if needed
            if (
                indx != 0
                and indx % self.accumulate_steps == 0
                and (indx // self.accumulate_steps) % self.args.validation_interval
                == (
                    self.args.validation_interval // 3
                )  # so we don't ovelap with checkpoint
            ):
                val_metrics = {}
                score = None

                if self.args.validate:
                    val_score, metrics = valid_step(self.state)
                    score = val_score
                    val_metrics.update(metrics)
                if self.args.evaluate and self.inference is not None:
                    self.inference.state = self.state
                    eval_metrics = self.inference.evaluate()
                    if len(eval_metrics) > 0:
                        eval_score = sum(eval_metrics.values()) / len(eval_metrics)
                        if score is None:
                            score = eval_score
                        else:
                            score = (score + eval_score) / 2
                        val_metrics.update(eval_metrics)
                if score is None:
                    score = float("-inf")

                val_metrics["train/tokens"] = (
                    ((indx + 1) // self.accumulate_steps)
                    * self.args.batch_size
                    * self.args.block_size
                )
                if self.main_process():
                    wandb.log(
                        val_metrics,
                        step=indx // self.accumulate_steps,
                    )
                    logger.info(
                        "VAL | {} | score {}",
                        indx // self.accumulate_steps,
                        score,
                    )

                if score > self.best_val_score_:
                    if self.main_process():
                        logger.info("VAL | BEST SCORE | score {}", score)
                    self.best_val_score_ = score
                    self.save(Path("best"))

        # final save at the end of training
        self.save(
            Path("final")
            / str(
                (
                    ((indx + 1) // self.accumulate_steps)
                    * self.args.batch_size
                    * self.args.block_size
                )
            )
        )

    def save(self, suffix: Path) -> None:
        """final save at the end of training"""

        self.save_tree_and_metadata(
            suffix,
            self.state,
            {"steps": self.global_step_counter_, "score": self.best_val_score_},
        )

        # Register as latest checkpoint for idempotent dispatch
        self.register(suffix)

        if self.main_process():
            logger.info(
                "CHECKPOINT | saved checkpoint at {} at step {}, best score {}",
                suffix,
                self.global_step_counter_,
                self.best_val_score_,
            )

    def load(self, suffix: Path) -> None:
        """load from a checkpoint, if available"""

        state, metadata = self.get_tree_and_metadata(suffix, self.state)

        self.state = state
        self.global_step_counter_ = metadata.get("steps", 0)
        self.best_val_score_ = metadata.get("score", float("-inf"))

        if self.main_process():
            logger.info(
                "CHECKPOINT | loaded checkpoint from {} at step {}, best score {}",
                suffix,
                self.global_step_counter_,
                self.best_val_score_,
            )

    def restore(self, suffix: Path) -> None:
        return self.load(suffix)  # this is to satisfy the restore API

    def run(self) -> None:
        """main entry point to run training, called on all nodes"""
        self.train()

    @property
    def done(self) -> bool:
        """check if training is done"""

        return self.global_step_counter_ >= self.total_batches
