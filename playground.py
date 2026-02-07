import jax
from transformers import AutoModelForCausalLM
import torchax as tx

import torch

# added later
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

from theseus.base.hardware import local
from theseus.base.topology import Topology

hardware = local("/sailhome/houjun/theseus", "../")
topo = Topology.new(hardware.chip)


# print(model_inputs)


# mesh = topo.mesh
# env = tx.default_env()


def output_flatten(v):
    return v.to_tuple(), None


def output_unflatten(aux, children):
    return modeling_outputs.CausalLMOutputWithPast(*children)


register_pytree_node(
    modeling_outputs.CausalLMOutputWithPast,
    output_flatten,
    output_unflatten,
)

from flax import linen as nn
import flax
from typing import Any


class Lmfmao(nn.Module):
    id: Any
    meta_model: Any = None

    def setup(self):
        if not self.has_variable("params", "_params"):
            _base = AutoModelForCausalLM.from_pretrained(self.id, dtype="bfloat16")
            with tx.default_env():
                _base.to("jax")
                params = dict(_base.named_parameters())
                buffers = dict(_base.named_buffers())

                params_jax = jax.tree_util.tree_map(lambda x: x.jax(), params)
                buffers_jax = jax.tree_util.tree_map(lambda x: x.jax(), buffers)

                self._params = self.param("_params", lambda *_: dict(params_jax))
                self.variable("buffers", "_buffers", lambda: dict(buffers_jax))
                _base.to("meta")
        else:
            self._params = self.get_variable("params", "_params")

    def __call__(self, x, attention_mask=None):
        if self.meta_model is None:
            raise ValueError("you must metamodel")

        x = tx.tensor.Tensor(x, tx.default_env())
        if attention_mask is not None:
            attention_mask = tx.tensor.Tensor(attention_mask, tx.default_env())
        params = jax.tree_util.tree_map(
            lambda x: tx.tensor.Tensor(x, tx.default_env()), self._params
        )
        buffer_state = self.get_variable("buffers", "_buffers")
        if buffer_state is None:
            raise ValueError("missing buffers/_buffers state")
        buffers = jax.tree_util.tree_map(
            lambda x: tx.tensor.Tensor(x, tx.default_env()), buffer_state
        )

        if isinstance(params, flax.core.FrozenDict):
            params = flax.core.frozen_dict.unfreeze(params)
        if isinstance(buffers, flax.core.FrozenDict):
            buffers = flax.core.frozen_dict.unfreeze(buffers)
        functional_tensors = dict(params)
        functional_tensors.update(buffers)

        with tx.default_env():
            (logits,) = torch.func.functional_call(
                self.meta_model,
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
            return logits.jax()


import jax.numpy as jnp

MODEL = "meta-llama/Llama-2-7b-hf"
meta_model = AutoModelForCausalLM.from_pretrained(MODEL)
meta_model = meta_model.to("cpu")

model = Lmfmao(MODEL, meta_model)
variables = model.init(jax.random.PRNGKey(7), jnp.ones((8, 4)).astype(jnp.int32))


# model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model_inputs = tokenizer(["Hello! What's the point of"], return_tensors="np")

@jax.jit
def apply_input(input_ids, variables):
    logits, buffer_updates = model.apply(
        variables, input_ids, mutable=["buffers"]
    )
    variables = flax.core.freeze(
        {**flax.core.unfreeze(variables), "buffers": buffer_updates["buffers"]}
    )

    return logits, variables

out, variables = apply_input(model_inputs["input_ids"], variables)
nn.with_partitioning(


# model.has_variable("params", "_params")
# # logits.shape

# # # variables

# def shard_weights_llama(mesh, weights):
#     weights = model.state_dict()
#     result = {}

#     for k, v in weights.items():
#         if (('q_proj.weight' in k) or
#             ('k_proj.weight' in k) or
#             ('v_proj.weight' in k) or
#             ('gate_proj.weight' in k) or
#             ('up_proj.weight' in k)):
#             sharding = P('shard', None)
#         elif(('o_proj.weight' in k) or
#             ('down_proj.weight' in k) or
#             ('lm_head.weight' in k) or
#             ('embed_tokens' in k)):
#             sharding = P(None, 'shard')
#         else:
#             sharding = P() # replicated

#         result[k] = v.apply_jax(jax.device_put, NamedSharding(mesh, sharding))


#     return result

