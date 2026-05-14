from dataclasses import dataclass
from typing import Any, List

import flax
import jax
import numpy as np
import optax
from flax.training import train_state
from jax import random as jax_random
from loguru import logger

from theseus.base import ExecutionSpec, PyTree
from theseus.config import configure, field
from theseus.data.tokenizer import get_tokenizer
from theseus.inference.base import InferenceJob
from theseus.model.models.base import GPT
from theseus.model.module import Module
from theseus.registry import job
from theseus.training.backbone import BACKBONES, BackboneConfig, ModelDtypeConfig


@dataclass
class ChatConfig:
    block_size: int = field("architecture/block_size", default=512)
    per_device_batch_size: int = field("training/per_device_batch_size", default=-1)
    max_new_tokens: int = field("inference/max_new_tokens", default=256)
    temperature: float = field("inference/temperature", default=0.8)
    top_p: float = field("inference/top_p", default=0.9)


@job("gpt/debug/chat/continuation")
class Chat(InferenceJob[ChatConfig, GPT]):
    MODEL = GPT

    @classmethod
    def config(cls) -> List[Any]:
        return [ChatConfig]

    def run(self) -> None:
        tok = get_tokenizer()

        max_new = self.args.max_new_tokens
        temp = self.args.temperature
        top_p = self.args.top_p

        logger.info(
            "CHAT | ready  max_new_tokens={}  temperature={}  top_p={}",
            max_new,
            temp,
            top_p,
        )
        print("\n--- chat (ctrl-c to quit) ---\n")

        eot = tok.eot_token

        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not prompt.strip():
                continue

            [gen_ids] = self.rollout(
                [prompt],
                tok,
                max_new_tokens=max_new,
                temperature=temp,
                top_p=top_p,
                return_type="output_indices",
            )

            if eot in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eot)]

            print(tok.decode(gen_ids))
            print()


@job("backbone/debug/chat/continuation")
class ChatPretrained(Chat):
    """Chat with a pretrained HuggingFace backbone loaded via from_pretrained."""

    MODEL = Module  # actual model class chosen at runtime via BACKBONES

    @classmethod
    def config(cls) -> List[Any]:
        return [ChatConfig, BackboneConfig, ModelDtypeConfig]

    def __init__(self, spec: ExecutionSpec) -> None:
        super().__init__(spec)

        assert spec.topology is not None, "Topology required for chat"
        self.mesh = spec.topology.mesh
        self.replicas = spec.topology.replicas
        self.local_replicas = spec.topology.local_replicas

        backbone_cfg = configure(BackboneConfig)
        dtype_cfg = configure(ModelDtypeConfig)

        model_cls = BACKBONES[backbone_cfg.implementation]
        self.model, params = model_cls.from_pretrained(
            backbone_cfg.weights,
            param_dtype=dtype_cfg.param_dtype,
            activation_dtype=dtype_cfg.activation_dtype,
        )

        self.key, self.dropout_key = jax_random.split(self.key)

        params = self._cast_params(params)

        def make_state(p: PyTree[Any]) -> train_state.TrainState:
            return train_state.TrainState.create(  # type: ignore[no-untyped-call]
                apply_fn=self.model.apply, params=p, tx=optax.identity()
            )

        state_shapes = jax.eval_shape(make_state, params)
        self.state_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore[attr-defined]
            flax.linen.get_partition_spec(state_shapes),
            self.mesh,
            rules=tuple(self.model.sharding),
        )
        self.state = jax.jit(make_state, out_shardings=self.state_sharding)(params)

        self.per_device_batch_size = self.args.per_device_batch_size
        self.block_size = self.args.block_size

    def _cast_params(self, params: PyTree[Any]) -> PyTree[Any]:
        target = np.dtype(self.model.param_dtype)

        def cast_leaf(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x.astype(target, copy=False)
            if isinstance(x, jax.Array):
                return x.astype(target)
            return x

        return jax.tree_util.tree_map(cast_leaf, params)
