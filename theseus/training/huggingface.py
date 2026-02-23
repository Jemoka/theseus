"""
Trainer helpers for HuggingFace-compatible Theseus models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar, cast

import flax
import flax.linen
import jax
import jax.numpy as jnp
from flax.training import train_state
from jax import random as jax_random

from theseus.base import PyTree
from theseus.config import configure
from theseus.evaluation.base import Evaluator
from theseus.model.huggingface import HFCompat
from theseus.training.trainer import BaseTrainer, BaseTrainerConfig

HM = TypeVar("HM", bound=HFCompat)


@dataclass
class HFTrainerConfig(BaseTrainerConfig):
    pass


class HFTrainState(train_state.TrainState):  # type: ignore[no-untyped-call]
    buffers: Any


class HFTrainer(BaseTrainer[HFTrainerConfig, HM], Generic[HM]):
    MODEL: Type[HM]
    CONFIG: Type[HFTrainerConfig] = HFTrainerConfig

    def evaluator(self) -> Evaluator[HM]:
        from theseus.evaluation.huggingface import HFEvaluator

        return cast(Evaluator[HM], HFEvaluator.from_trainer(self))

    def _init_model(self) -> PyTree[jax.Array]:
        self.model = configure(self.MODEL)
        self.key, init_key, self.dropout_key = jax_random.split(self.key, num=3)

        dummy_input = jnp.ones((1, self.args.block_size), dtype=jnp.int32)
        variables, _ = self.sharded_init(
            self.model, init_key, dummy_input, mesh=self.mesh
        )
        self._buffers: Any = variables.get("buffers", flax.core.freeze({}))
        return cast(PyTree[jax.Array], variables["params"])

    def _init_optimizer(self, params: PyTree[jax.Array]) -> None:
        self.scheduler = self._schedule()
        self.tx = self._optimizer()

        buffers = self._buffers

        def make_state(p: PyTree[jax.Array]) -> HFTrainState:
            return cast(
                HFTrainState,
                HFTrainState.create(  # type: ignore[no-untyped-call]
                    apply_fn=self.model.apply, params=p, tx=self.tx, buffers=buffers
                ),
            )

        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
            flax.linen.get_partition_spec(state_shapes),
            self.mesh,
            rules=tuple(self.model.sharding),
        )
        self.state = cast(
            train_state.TrainState,
            jax.jit(make_state, out_shardings=self.state_sharding)(params),
        )
        self.total_params = (
            sum(x.size for x in jax.tree_util.tree_leaves(self.state.params)) / 1e6
        )

    @staticmethod
    def _forward_with_buffers(
        state: train_state.TrainState,
        params: Any,
        buffers: Any,
        batch: Tuple[jax.Array, jax.Array, jax.Array],
        mutable_buffers: bool,
    ) -> Tuple[jax.Array, jax.Array, Any]:
        x, y, padding_mask = batch

        variables = {"params": params, "buffers": buffers}
        if mutable_buffers:
            (logits, loss), updates = state.apply_fn(
                variables,
                x,
                y,
                padding_mask=padding_mask,
                mutable=["buffers"],
            )
            new_buffers = updates.get("buffers", buffers)
        else:
            logits, loss = state.apply_fn(
                variables,
                x,
                y,
                padding_mask=padding_mask,
            )
            new_buffers = buffers

        return logits, loss, new_buffers

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
        del key, deterministic, extra_variables
        from typing import cast as type_cast

        xb: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], batch)
        x = xb["x"]
        y = xb["y"]
        padding_mask = xb["padding_mask"]
        buffers = getattr(state, "buffers", flax.core.freeze({}))
        logits, loss, _ = HFTrainer._forward_with_buffers(
            state=state,
            params=params,
            buffers=buffers,
            batch=(x, y, padding_mask),
            mutable_buffers=False,
        )
        if mutable is not None:
            return (logits, loss, {}), flax.core.freeze({})
        return logits, loss, {}

    @classmethod
    def train_step(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],
        key: jax.Array,
        accumulate_steps: int,
    ) -> Tuple[train_state.TrainState, jax.Array, Any]:
        del key

        def train_eval(
            state: train_state.TrainState,
            batch: PyTree[jax.Array],
            buffers: Any,
        ) -> Tuple[jax.Array, PyTree[jax.Array], Any, Any]:
            from typing import cast as type_cast

            xb: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], batch)
            x = xb["x"]
            y = xb["y"]
            padding_mask = xb["padding_mask"]

            def loss_fn(
                params: PyTree[jax.Array], buffers_inner: Any
            ) -> Tuple[jax.Array, Any]:
                _, loss, new_buffers = cls._forward_with_buffers(
                    state,
                    params,
                    buffers_inner,
                    (x, y, padding_mask),
                    mutable_buffers=True,
                )
                return loss / accumulate_steps, new_buffers

            (loss, new_buffers), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, buffers
            )
            return loss, grads, new_buffers, {}

        def reduce(
            carry: Tuple[PyTree[jax.Array], jax.Array, Any],
            batch_item: Any,  # PyTree with single batch item
        ) -> Tuple[Tuple[PyTree[jax.Array], jax.Array, Any], Any]:
            grad, loss, buffers = carry
            loss_single, grad_single, buffers_next, meta = train_eval(
                state, batch_item, buffers
            )

            grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad, grad_single)
            loss_acc = loss + loss_single
            return (grad_acc, loss_acc, buffers_next), meta

        grad_zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        buffers0 = getattr(state, "buffers", flax.core.freeze({}))
        (grad_sum, loss_sum, buffers_out), metas = jax.lax.scan(
            reduce,
            (grad_zero, jnp.array(0.0), buffers0),
            batch,
        )
        last_meta: Any = jax.tree_util.tree_map(lambda x: x[-1], metas)
        new_state = state.apply_gradients(  # type: ignore[no-untyped-call]
            grads=grad_sum, buffers=buffers_out
        )
        return cast(train_state.TrainState, new_state), loss_sum, last_meta

    @classmethod
    def val_step(
        cls,
        state: train_state.TrainState,
        batch: PyTree[jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        from typing import cast as type_cast

        batch_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], batch)
        x = batch_dict["x"]
        y = batch_dict["y"]
        padding_mask = batch_dict["padding_mask"]

        def reduce(
            carry: Tuple[jax.Array, jax.Array],
            xb_item: Any,  # PyTree with single batch item
        ) -> Tuple[Tuple[jax.Array, jax.Array], None]:
            loss_sum, count = carry
            xb_dict: Dict[str, jax.Array] = type_cast(Dict[str, jax.Array], xb_item)
            x_i = xb_dict["x"]
            y_i = xb_dict["y"]
            mask_i = xb_dict["padding_mask"]

            _, loss_i, _ = cls._forward_with_buffers(
                state,
                state.params,
                getattr(state, "buffers", flax.core.freeze({})),
                (x_i, y_i, mask_i),
                mutable_buffers=False,
            )
            n = mask_i.sum()
            return (loss_sum + loss_i * n, count + n), None

        # Create dict of arrays to scan over
        batch_to_scan = {"x": x, "y": y, "padding_mask": padding_mask}
        (loss_sum, count), _ = jax.lax.scan(
            reduce, (jnp.array(0.0), jnp.array(0)), batch_to_scan
        )
        return loss_sum, count
