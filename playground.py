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

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-hf",
#     dtype="bfloat16"
# )

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


# # jax array inside
# with env:
#   model.to('jax')
#   weights = shard_weights_llama(mesh, model.state_dict())


# with env:
#     def
#     input_ids = model_inputs.input_ids.to('jax').apply_jax_(
#         jax.device_put,
#         NamedSharding(mesh, P()))

#     attention_mask = model_inputs.attention_mask.to('jax').apply_jax_(
#         jax.device_put,
#         NamedSharding(mesh, P())
#     )

#     (logits, ) = torch.func.functional_call(
#         model,
#         weights,
#         (input_ids,),
#         dict(
#             attention_mask=attention_mask,
#             return_dict=False,
#             use_cache=False,
#         ),
#     )

# with env:
#     rs = logits.argmax(dim=-1)
#     decoded = rs.cpu()

# tokenizer.decode(decoded)


# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-hf",
#     dtype="meta"
# )
# state_dict =

from flax import linen as nn
import flax
from typing import Any


class Lmfmao(nn.Module):
    id: Any
    meta_model: Any = None

    def setup(self):
        if not self.has_variable("param", "_params"):
            _base = AutoModelForCausalLM.from_pretrained(self.id, dtype="bfloat16")
            with tx.default_env():
                _base.to("jax")
                sd = dict(_base.state_dict())
                sd_jax = jax.tree_util.tree_map(lambda x: x.jax(), sd)
                self._params = self.param("_params", lambda *_: dict(sd_jax))
                _base.to("meta")
        else:
            self._params = self.get_variable("param", "_params")

    def __call__(self, x, attention_mask=None):
        if self.meta_model is None:
            raise ValueError("you must metamodel")

        x = tx.tensor.Tensor(x, tx.default_env())
        if attention_mask is not None:
            attention_mask = tx.tensor.Tensor(attention_mask, tx.default_env())
        params = jax.tree_util.tree_map(
            lambda x: tx.tensor.Tensor(x, tx.default_env()), self._params
        )

        with tx.default_env():
            (logits,) = torch.func.functional_call(
                self.meta_model,
                flax.core.frozen_dict.unfreeze(params),
                (x,),
                dict(
                    attention_mask=attention_mask,
                    return_dict=False,
                    use_cache=False,
                ),
            )
        return logits


import jax.numpy as jnp

MODEL = "meta-llama/Llama-2-7b-hf"
meta_model = AutoModelForCausalLM.from_pretrained(MODEL)
meta_model = meta_model.to("cpu")

model = Lmfmao(MODEL, meta_model)
variables = model.init(jax.random.PRNGKey(7), jnp.ones((8, 4)).astype(jnp.int32))

model
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model_inputs = tokenizer(["help me sober up please"], return_tensors="np")

# logits = model.apply(variables, model_inputs["input_ids"])
# logits.shape

# # variables
