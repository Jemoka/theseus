"""
Just a humble ~~polytime verifier~~ pretrained evaluator.
"""

import random as py_random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import flax
import jax
import numpy as np
import optax
from flax.training import train_state
from jax import random as jax_random

from theseus.base import ExecutionSpec, PyTree
from theseus.config import configure, field
from theseus.data.tokenizer import get_tokenizer
from theseus.evaluation.base import Evaluator
from theseus.model.models import GPT
from theseus.model.module import Module
from theseus.registry import EVALUATIONS, job
from theseus.training.backbone import BACKBONES, BackboneConfig, ModelDtypeConfig


@dataclass
class _EvalSizingConfig:
    """Sizing knobs the evaluator needs from outside ``EvaluatorConfig``.

    ``InferenceJob.restore_from_path`` reads these from the checkpoint's
    config; for the ``from_pretrained`` path we hydrate them ourselves.
    """

    block_size: int = field("architecture/block_size", default=512)
    per_device_batch_size: int = field("training/per_device_batch_size", default=1)


@job("gpt/evaluate")
class Evaluate(Evaluator[GPT]):
    """Run the configured evaluation suite against a GPT checkpoint.

    Invoke with ``--restore <ckpt>``; the checkpoint's state is loaded via
    ``InferenceJob.restore_from_path`` and the evaluator-specific fields
    (encoding / evaluations / sampling rng) are wired up on top.
    """

    MODEL = GPT

    def restore_from_path(self, rel_path: str | Path) -> None:
        super().restore_from_path(rel_path)
        self._init_evaluator()

    def _init_evaluator(self) -> None:
        self.encoding = get_tokenizer()
        self.length = self.args.length
        self.random = py_random.Random(0xC0FFEE)
        try:
            self.evaluations = [EVALUATIONS[name]() for name in self.args.components]
        except KeyError as e:
            raise ValueError(f"Unknown evaluation dataset: {e.args[0]}") from e


@job("backbone/evaluate")
class BackboneEvaluate(Evaluate):
    """Run the configured evaluation suite against a HuggingFace backbone."""

    MODEL = Module  # type: ignore[assignment]

    @classmethod
    def config(cls) -> List[Any]:
        return super().config() + [BackboneConfig, ModelDtypeConfig, _EvalSizingConfig]

    def __init__(self, spec: ExecutionSpec) -> None:
        super().__init__(spec)

        assert spec.topology is not None, "Topology required for evaluation"
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas

        backbone_cfg = configure(BackboneConfig)
        dtype_cfg = configure(ModelDtypeConfig)
        sizing = configure(_EvalSizingConfig)

        model_cls = BACKBONES[backbone_cfg.implementation]
        self.model, params = model_cls.from_pretrained(
            backbone_cfg.weights,
            param_dtype=dtype_cfg.param_dtype,
            activation_dtype=dtype_cfg.activation_dtype,
        )

        self.key, self.dropout_key = jax_random.split(self.key)

        params = self._cast_params(params)

        def make_state(p: PyTree[Any]) -> train_state.TrainState:
            return train_state.TrainState.create(  # type: ignore[no-untyped-call, no-any-return]
                apply_fn=self.model.apply, params=p, tx=optax.identity()
            )

        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
            flax.linen.get_partition_spec(state_shapes),
            self.mesh,
            rules=tuple(self.model.sharding),
        )
        self.state = jax.jit(make_state, out_shardings=self.state_sharding)(params)

        self.per_device_batch_size = sizing.per_device_batch_size
        self.block_size = sizing.block_size

        self._init_evaluator()

    def _cast_params(self, params: PyTree[Any]) -> PyTree[Any]:
        target = np.dtype(self.model.param_dtype)

        def cast_leaf(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x.astype(target, copy=False)
            if isinstance(x, jax.Array):
                return x.astype(target)
            return x

        return jax.tree_util.tree_map(cast_leaf, params)
