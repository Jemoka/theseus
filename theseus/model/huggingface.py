"""
huggingface compatibility layer
I can't believe this is not butter.
"""

from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Optional, Self, Type, TypeAlias

import flax
import jax
import jax.numpy as jnp

import torch
import torchax as tx
from jax.tree_util import DictKey, GetAttrKey, SequenceKey, register_pytree_node
from transformers import AutoConfig, AutoModelForCausalLM, modeling_outputs

try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights

from theseus.model.axes import Axes
from theseus.model.module import Module


LogicalAxes: TypeAlias = tuple[Optional[Axes], ...]


_META_MODEL_CACHE_MAX_SIZE = 16
_META_MODEL_CACHE: OrderedDict[str, Any] = OrderedDict()


def _cache_meta_model(model_id: str, model: Any) -> Any:
    _META_MODEL_CACHE[model_id] = model
    _META_MODEL_CACHE.move_to_end(model_id)
    while len(_META_MODEL_CACHE) > _META_MODEL_CACHE_MAX_SIZE:
        _META_MODEL_CACHE.popitem(last=False)
    return model


def _build_meta_model_cached(model_id: str) -> Any:
    cached = _META_MODEL_CACHE.get(model_id)
    if cached is not None:
        _META_MODEL_CACHE.move_to_end(model_id)
        return cached

    model_config = AutoConfig.from_pretrained(model_id)
    with no_init_weights():
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                model_config, torch_dtype=torch.bfloat16
            )
    return _cache_meta_model(model_id, model)


def output_flatten(
    v: modeling_outputs.CausalLMOutputWithPast,
) -> tuple[tuple[Any, ...], None]:
    return v.to_tuple(), None


def output_unflatten(
    aux: None, children: tuple[Any, ...]
) -> modeling_outputs.CausalLMOutputWithPast:
    return modeling_outputs.CausalLMOutputWithPast(*children)


# register the CausalLMOutputWithPast as a pytree node so that it can be used in JAX transformations
try:
    register_pytree_node(
        modeling_outputs.CausalLMOutputWithPast,
        output_flatten,
        output_unflatten,
    )
except ValueError as e:
    if "Duplicate" not in str(e):
        raise e


class HFCompat(Module):
    id: str

    @classmethod
    @abstractmethod
    def axes(cls, x: str) -> Optional[LogicalAxes]:
        raise NotImplementedError

    @classmethod
    def components(cls) -> list[Type[Any]]:
        return []

    @classmethod
    def _build_meta_model(cls, model_id: str) -> Any:
        return _build_meta_model_cached(model_id)

    @staticmethod
    def _partition_names(axes: LogicalAxes) -> tuple[Optional[str], ...]:
        return tuple(axis.value if axis is not None else None for axis in axes)

    @staticmethod
    def _param_name_from_path(path: tuple[Any, ...]) -> str:
        leaf = path[-1]
        if isinstance(leaf, DictKey):
            return str(leaf.key)
        if isinstance(leaf, GetAttrKey):
            return str(leaf.name)
        if isinstance(leaf, SequenceKey):
            return str(leaf.idx)
        return str(leaf)

    def _partition_parameter(self, path: tuple[Any, ...], x: Any) -> Any:
        maybe_axes = self.axes(self._param_name_from_path(path))
        if maybe_axes is None:
            return x
        return flax.linen.Partitioned(x, self._partition_names(maybe_axes))

    @staticmethod
    def _to_tx_tensor(x: Any) -> Any:
        while isinstance(x, flax.core.meta.Partitioned):
            x = x.value
        if isinstance(x, tx.tensor.Tensor):
            return x
        return tx.tensor.Tensor(x, tx.default_env())

    def setup(self) -> None:
        if not self.has_variable("params", "_params"):
            _base = AutoModelForCausalLM.from_pretrained(
                self.id, torch_dtype=torch.bfloat16
            )
            with tx.default_env():
                _base.to("jax")
                params = dict(_base.named_parameters())
                buffers = dict(_base.named_buffers())

                params_jax = jax.tree_util.tree_map(lambda x: x.jax(), params)
                buffers_jax = jax.tree_util.tree_map(lambda x: x.jax(), buffers)
                params_jax = dict(params_jax)

                # map parameter names to linen.Partitioned using self.axes()
                params_jax = jax.tree_util.tree_map_with_path(
                    self._partition_parameter,
                    params_jax,
                )

                self._params = self.param("_params", lambda *_: params_jax)
                self.variable("buffers", "_buffers", lambda: dict(buffers_jax))
                _base.to("meta")  # free up memory
                _cache_meta_model(self.id, _base)
        else:
            self._params = self.get_variable("params", "_params")

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        """Compute cross-entropy loss given logits and targets."""

        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)

        # Mask out ignore index (-1)
        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)
        vocab_size = logits_flat.shape[-1]

        loss = -jnp.sum(
            jax.nn.log_softmax(logits_flat, axis=-1)
            * jax.nn.one_hot(targets_masked, vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)

        return loss

    def __call__(
        self,
        x: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        meta_model = self._build_meta_model(self.id)

        x = tx.tensor.Tensor(x, tx.default_env())
        attention_mask: Optional[Any] = None
        if padding_mask is not None:
            attention_mask = tx.tensor.Tensor(
                padding_mask.astype(jnp.bool_), tx.default_env()
            )
        params = jax.tree_util.tree_map(
            self._to_tx_tensor,
            self._params,
            is_leaf=lambda leaf: isinstance(leaf, flax.core.meta.Partitioned),
        )
        buffer_state: Optional[Any] = self.get_variable("buffers", "_buffers")
        if buffer_state is None:
            raise ValueError("missing buffers/_buffers state")
        buffers = jax.tree_util.tree_map(
            self._to_tx_tensor,
            buffer_state,
            is_leaf=lambda leaf: isinstance(leaf, flax.core.meta.Partitioned),
        )

        if isinstance(params, flax.core.FrozenDict):
            params = flax.core.frozen_dict.unfreeze(params)
        if isinstance(buffers, flax.core.FrozenDict):
            buffers = flax.core.frozen_dict.unfreeze(buffers)
        functional_tensors = dict(params)
        functional_tensors.update(buffers)

        with tx.default_env():
            (logits,) = torch.func.functional_call(
                meta_model,
                functional_tensors,
                (x,),
                dict(
                    attention_mask=attention_mask,
                    return_dict=False,
                    use_cache=False,
                ),
            )

            if self.is_mutable_collection("buffers"):
                self.put_variable(
                    "buffers",
                    "_buffers",
                    {
                        name: tensor.jax()
                        for name, tensor in functional_tensors.items()
                        if name in buffer_state
                    },
                )

            logits_jax = logits.jax()
            if targets is not None:
                loss = self.loss(logits_jax, targets)
                return logits_jax, loss
            else:
                return logits_jax, None

    @classmethod
    def from_pretrained(cls, model_id: str) -> Self:
        cls._build_meta_model(model_id)
        return cls(id=model_id)


# if __name__ == "__main__":
#     import jax.numpy as jnp
#     from transformers import AutoTokenizer

#     MODEL = "meta-llama/Llama-2-7b-hf"
#     model_config = AutoConfig.from_pretrained(MODEL)
#     with no_init_weights():
#         with torch.device("meta"):
#             meta_model = AutoModelForCausalLM.from_config(
#                 model_config, dtype=torch.bfloat16
#             )

#     test = HFCompat(id=MODEL, meta_model=meta_model)
#     variables = test.init(jax.random.PRNGKey(7), jnp.ones((8, 4)).astype(jnp.int32))

#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     model_inputs = tokenizer(["Hello! What's the point of"], return_tensors="np")

#     logits, buffer_updates = test.apply(
#         variables, model_inputs["input_ids"], mutable=["buffers"]
#     )
#     variables = flax.core.freeze(
#         {**flax.core.unfreeze(variables), "buffers": buffer_updates["buffers"]}
#     )
