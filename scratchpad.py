import os
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
    "<level>{level: ^8}</level>| "
    "<magenta>({name}:{line})</magenta> <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    enqueue=True,
    filter=lambda x: x["extra"].get("task", "") != "plot",
)
#
# !ls $HF_HOME/hub

import jax
import jax.numpy as jnp

from theseus.mock import Mocker
self = Mocker()



# from theseus.config import *
# from theseus.quick import init

# from theseus.experiments.models.gpt import PretrainGPT


# import torch
# from transformers import Qwen2ForCausalLM

# import jax
# import jax.numpy as jnp
# import jax.nn as jnn
# import flax.linen as nn

# # from theseus.experiments.models.forking import PretrainThoughtbubbles
# # from theseus.experiments.continual.abcd import ABCDTrainer
# # from theseus.evaluation.base import Evaluator
# # from theseus.training.backbone import BackbonedTrainer
# from theseus.registry import JOBS

# from omegaconf import OmegaConf


# # q.config.training.per_device_batch_size = 2
# # j = q.create()
# # config = q.config

# from theseus.mock import Mocker
# self = Mocker()

# from random import Random

# N_FUNCTIONS = 32
# N_VALUES = 1024
# FIXED_SEED = 7
# MAX_SEQ_LENGTH = 512
# SEQUENCES = 100000

# R = Random(FIXED_SEED)

# function_tokens = [i+1 for i in range(N_FUNCTIONS)]
# value_tokens = [i + N_FUNCTIONS+1 for i in range(N_VALUES)]

# # build functions
# functions = []
# for indx in range(len(function_tokens)):
#     func = {}
#     for i in value_tokens:
#         target = R.choice(value_tokens)
#         func[i] = target
#     functions.append(func)

# # special tokesn
# START_TOKEN = N_FUNCTIONS + N_VALUES + 1
# SEP_TOKEN = START_TOKEN + 1
# EOT_TOKEN = 0

# # compute vocab size
# VOCAB_SIZE = SEP_TOKEN + 1

# # build sequences
# sequences = []
# for _ in range(SEQUENCES):
#     # template: should be 
#     # f1 f2 f3 f4 <start> v1 <sep> f4(f3(f2(f1(v1))))
#     # the idea is that this is the "least"
#     # efficient representation of a sequence, since the
#     # v1 itself doesn't tell you anything about the
#     # 

#     seq = []
#     for _ in range((MAX_SEQ_LENGTH-2)):
#         f = R.choice(function_tokens)
#         seq.append(f)
#     init_value = R.choice(value_tokens)
#     for f in seq:
#         init_value = functions[f-1][init_value]
#     seq.append(START_TOKEN)
#     seq.append(init_value)
#     seq.append(SEP_TOKEN)
#     seq.append(init_value)
#     sequences.append(seq)

# # to check, we will just .split(" ") the last element
# # and then chechk the tokens. nice thing is that given
# # a configuration of the dataset we can just
# # use the same seed and regenerate the same dataset, which is nice for testing
        

    
    

        

    





from omegaconf import OmegaConf
from theseus.quick import init

self = Mocker()
cfg = OmegaConf.load("./configs/gpt/small.yaml")
q = init("gpt/train/pretrain", "test", config=cfg)

q.config.logging.wandb = False
q.config.training.per_device_batch_size = 4
q.config.logging.report_interval = 1
q.config.logging.validation_interval = 2
# job = q.create()
# # self.model = job.model


# # x = jnp.ones((4, q.config.architecture.block_size), dtype=jnp.int32)
# # y = jnp.ones((4, q.config.architecture.block_size), dtype=jnp.int32)
# # padding_mask = jnp.ones((4, q.config.architecture.block_size), dtype=bool)
# # intermediates = self.model.intermediates(x,y,padding_mask)

# # ### plot embeddings ###
# # embeddings = [intermediates["plots"]["embeddings"]] + [
# #     i["embeddings"]
# #     for i in intermediates["plots"].values()
# #     if isinstance(i, dict)
# # ]
# # max_seq_len = max([e[0].shape[1] for e in embeddings])
# # padded_embeddings = []

# # for i in embeddings:
# #     e = i[0]
# #     if e.shape[1] < max_seq_len:
# #         pad_width = ((0, 0), (0, max_seq_len - e.shape[1]), (0, 0))
# #         e = jnp.pad(e, pad_width)
# #     padded_embeddings.append(e)
# # stacked_embeddings = jnp.array(padded_embeddings)
# # embeddings_2d = stacked_embeddings[:, 0]  # only one sample batch axes

# # embeddings_2d.shape


# # def vectors_to_colors(vecs_2d):
# #     """(layers, seq_len, 2) -> (layers, seq_len, 3) RGB in [0,1]."""
# #     shape = vecs_2d.shape[:2]
# #     flat = vecs_2d.reshape(-1, 2)

# #     angle = np.arctan2(flat[:, 1], flat[:, 0])          # [-π, π]
# #     radius = np.linalg.norm(flat, axis=-1)

# #     hue = (angle + np.pi) / (2 * np.pi)                 # [0, 1]

# #     # percentile-clip radius so outliers don't wash everything out
# #     lo, hi = np.percentile(radius, [2, 98])
# #     sat = 0.3 + 0.7 * np.clip((radius - lo) / (hi - lo + 1e-8), 0, 1)

# #     val = np.full_like(hue, 0.92)

# #     hsv = np.stack([hue, sat, val], axis=-1)
# #     rgb = mcolors.hsv_to_rgb(hsv)
# #     return rgb.reshape(*shape, 3)


# # vectors_to_colors(embeddings_2d.astype(jnp.float32))


# # # intermediates["plots"]["blocks_3"]["ssca"]["scratching_attn_weights"][0]


# # # import seaborn as sns

# # # # weights = [
# # # #     i["ssca"]["scratching_attn_weights"][0][0][0]
# # # #     for i in intermediates["plots"].values()
# # # #     if isinstance(i, dict)
# # # # ]
# # # # weights[0].shape # => 1024, 512
# # # # weights[1].shape # => 1024, 1024
# # # # max_seq_len = max([i.shape[-1] for i in weights])

# # # # # pad weights to max_seq_len so we can stack them
# # # # padded_weights = []
# # # # for w in weights:
# # # #     pad_width = max_seq_len - w.shape[-1]
# # # #     if pad_width > 0:
# # # #         w = jnp.pad(w, ((0, 0), (0, pad_width)), constant_values=-jnp.inf)
# # # #     padded_weights.append(w)

# # # # # aaand softmax
# # # # padded_weights = jax.nn.softmax(jnp.stack(padded_weights), axis=-1)

# # # # # plot
# # # # for i, w in enumerate(padded_weights):
# # # #     fig_1, ax = plt.subplots(figsize=(12, 6))
# # # #     sns.heatmap(w.astype(jnp.float32), ax=ax, cmap="viridis")

# # # #     ax.set_xlabel("token index (queries, forks)")
# # # #     ax.set_ylabel("token index (keys, seq)")
# # # #     ax.set_title(f"Attention Weights for Block {i}")
# # # #     plt.tight_layout()

# # # #     fig_1.savefig(f"./block_{i}_attention_weights.png")
# # # embeddings = [intermediates["plots"]["embeddings"]] + [
# # #     i["embeddings"]
# # #     for i in intermediates["plots"].values()
# # #     if isinstance(i, dict)
# # # ]
# # # max_seq_len = max([e[0].shape[1] for e in embeddings])
# # # padded_embeddings = []

# # # for i in embeddings:
# # #     e = i[0]
# # #     if e.shape[1] < max_seq_len:
# # #         pad_width = ((0, 0), (0, max_seq_len - e.shape[1]), (0, 0))
# # #         e = jnp.pad(e, pad_width)
# # #     padded_embeddings.append(e)
# # # stacked_embeddings = jnp.array(padded_embeddings)
# # # embeddings_2d = stacked_embeddings[:, 0]  # only one sample batch axes




# # # # # from theseus.config import configure
# # # # # self.sb = configure(Scratchbubbles)
# # # # q.config.training.per_device_batch_size = 2
# # # # q.config.training.batch_size = 16
# # # # q.config.training.evaluate = False
# # # # q.config.logging.wandb = False
# # # # # disable structarl and addt pltos
# # # # OmegaConf.set_struct(q.config, False)
# # # # q.config.logging.plots = OmegaConf.create({"save": True})
# # # # OmegaConf.set_struct(q.config, True)
# # # # # and then wee?
# # # # job = q()
# # # # 1+1



# # # # intermediates = self.sb.intermediates(x, y, padding_mask)

# # # # [i for i in intermediates["plots"].keys()]

# # # # self.model = j.model



# # # # # from theseus.mock import Mocker
# # # # # self = Mocker()
# # # # # self.model = j.model
# # # # # 1+1

# # # # # batch = j.batch()
# # # # # intermediates = self.model.intermediates(batch["x"],batch["y"],batch["padding_mask"])
# # # # # intermediates


# # # # # from matplotlib import pyplot as plt
# # # # # import seaborn as sns
# # # # # batch["x"].shape

# # # # # scores = [i["new_cumulative_scores"] for i in intermediates["plots"].values()]
# # # # # scores = jnp.stack(jnp.array(scores))
# # # # # scores = jnp.exp(scores[:, 0, 0])  # type: ignore
# # # # # scores = scores.at[scores > 1].set(1.0)
# # # # # scores = np.array(scores)
# # # # # scores = scores.astype(np.float32)

# # # # # fig = plt.figure(figsize=(10, 6))
# # # # # ax = fig.add_subplot(111)
# # # # # ax = sns.heatmap(scores, ax=ax)
# # # # # fig.savefig("wut.png")



# # # # # q.config.architecture.fork

# # # # # intermediates



# # # # # # # q.config.tokenizer.backend = "huggingface"
# # # # # # # q.config.tokenizer.name = "Qwen/Qwen2.5-0.5B"
# # # # # # job = q.create()
# # # # # # job()

# # # # # # evaluator = job.evaluator()
# # # # # # res = evaluator.rollout([
# # # # # #     "The Federal Reserve said last Tuesday that",
# # # # # #     "Robustness to transitioning is a big issue that Rhodesia has thought a lot about, since"
# # # # # # ], max_new_tokens=10, top_p=0.9, temperature=0.7)
# # # # # # res

# # # # # # # q.close()

# # # # # # from types import SimpleNamespace
# # # # # # self = SimpleNamespace()


# # # # # # res = jax.eval_shape(
# # # # # #     self.rms_1.apply, jnp.ones((8, q.config.architecture.block_size, q.config.architecture.n_embd))
# # # # # # )

# # # # # # import jax
# # # # # # def do_thing(x):
# # # # # #     vars = self.rms_1.init(jax.random.PRNGKey(0), x)
# # # # # #     self.rms_1.apply(vars,x)
# # # # # #     return x

# # # # # # jax.eval_shape(lambda:self.rms_1.init(jax.random.PRNGKey(0), tmp))
    
# # # # # # tmp = jax.ShapeDtypeStruct(.shape, jnp.float32)
# # # # # # x = jnp.ones((8, q.config.architecture.block_size, q.config.architecture.n_embd))
# # # # # # res = self.mlp(jnp.ones((8, q.config.architecture.block_size, q.config.architecture.n_embd)))
# # # # # # type(res)
# # # # # # self.rms_1.obj
# # # # # # res = jax.eval_shape(do_thing, tmp)
# # # # # # res

# # # # # # self = Mocker()
# # # # # # self.n_embd = 128

# # # # # # self.three = 32


# # # # # # def mock(module):

# # # # # # import flax.linen as nn

# # # # # # class SowCNN(nn.Module):
# # # # # #   @nn.compact
# # # # # #   def __call__(self, x):
# # # # # #     x = nn.Conv(features=32, kernel_size=(3, 3))(x)
# # # # # #     x = nn.relu(x)
# # # # # #     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
# # # # # #     x = nn.Conv(features=64, kernel_size=(3, 3))(x)
# # # # # #     x = nn.relu(x)
# # # # # #     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
# # # # # #     x = x.reshape((x.shape[0], -1))  # flatten
# # # # # #     self.sow('intermediates', 'features', x)
    
# # # # # #     x = nn.Dense(features=256)(x)
# # # # # #     x = nn.relu(x)
# # # # # #     x = nn.Dense(features=10)(x)
# # # # # #     x = nn.log_softmax(x)
# # # # # #     return x

# # # # # # import jax
# # # # # # import jax.numpy as jnp

# # # # # # variables = SowCNN().init(jax.random.PRNGKey(7), jnp.ones((1, 28, 28, 1)))
# # # # # # res, mod = SowCNN().apply(variables, jnp.ones((1, 28, 28, 1)), mutable='intermediates')
# # # # # # mod
                        
# # # # # #   # By default the 'intermediates' collection is not mutable during init.
# # # # # #   # So variables will only contain 'params' here.
# # # # # #   return variables

# # # # # # @jax.jit
# # # # # # def predict(variables, x):
# # # # # #   # If mutable='intermediates' is not specified, then .sow() acts as a noop.
# # # # # #   output, mod_vars = SowCNN2().apply(variables, x, mutable='intermediates')
# # # # # #   features = mod_vars['intermediates']['SowCNN']['features']
# # # # # #   return output, features



# # # # # # os.environ["WANDB_DISABLED"] = "true"
# # # # # # from theseus.base.job import ExecutionSpec
# # # # # # spec = ExecutionSpec.local(os.environ.get("THESEUS_ROOT", "."), "harden-qwen2_7b-test", "redcodegen", "e0")
# # # # # # job = Hardening.from_checkpoint("ntoks/62914560", spec)
# # # # # # from theseus.experiments.models.qwen import EvaluateQwen

# # # # # # from theseus.config import configuration

# # # # # # cfg = job[1]
# # # # # # cfg.tokenizer.backend = "huggingface"
# # # # # # with configuration(cfg):
# # # # # #     ev = EvaluateQwen.from_trainer(job[0])

# # # # # # from theseus.data.tokenizer import encode_chat_template, ChatTemplate, ChatTurn
# # # # # # tmp = encode_chat_template(
# # # # # #     [
# # # # # #         ChatTurn(
# # # # # #             role="system",
# # # # # #             message="""You are a helpful assistant for generating code. Given the prompt, generate Python code that solves the task; as a reminder, you should be writing production code (i.e. disable any debugging traces, etc.) Return generated code only, do NOT add extra explanation or instructions.""",
# # # # # #         ),
# # # # # #         ChatTurn(role="user", message="Implement a Flask application that tracks user interactions (such as page views and actions) by utilizing Flask-SQLAlchemy for session management. Ensure that the session data is preserved accurately throughout the user's activity and is not cleared unexpectedly.")
# # # # # #     ],
# # # # # #     ev.encoding, 
# # # # # #     prompt=True
# # # # # # )
# # # # # # inf = job[0].inference

# # # # # # with quick(Hardening, "test") as j:

# # # # # # #     j.config.architecture.backbone.implementation = "qwen"
# # # # # # #     j.config.architecture.backbone.weights = "Qwen/Qwen2.5-Coder-7B-Instruct"
# # # # # # #     j.config.logging.report_interval = 
# # # # # # #     j.config.architecture.block_size = 1024
# # # # # # #     j.config.training.per_device_batch_size = 1
# # # # # # #     j.config.training.batch_size = 32

# # # # # # #     # contrastive learning, yolo, eventually maybe should
# # # # # # #     # evaluate i.e. by literally rolling out the model
# # # # # # #     j.config.training.evaluate = False
# # # # # # #     j.config.training.validation = False

# # # # # # #     # tokenizer is qwen
# # # # # # #     j.config.tokenizer.backend = "huggingingface"
# # # # # # #     j.config.tokenizer.name = "Qwen/Qwen2.5-Coder-7B-Instruct"
# # # # # # #     j.config.training.dataset = [
# # # # # # #         {
# # # # # # #             "name": "redcodegen__hardening",
# # # # # # #             "suffix": "qwen2code7b",
# # # # # # #             "style": "CONTRASTIVE",
# # # # # # #             "rate": "1.0",
# # # # # # #         }
# # # # # # #     ]
# # # # # # #     # j.save("./configs/redcodegen/hardeningy.yaml", n_shards=2)
# # # # # # #     j()


# # # # # # # torch_dtype = torch.float32
# # # # # # # hf_model = Qwen2ForCausalLM.from_pretrained(
# # # # # # #     "Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype=torch_dtype, device_map=None
# # # # # # # )
# # # # # # # device = "cpu"
# # # # # # # hf_model.to(device)
# # # # # # # hf_model.eval()
# # # # # # # cfg = hf_model.config

# # # # # # # rope_theta = 10000.0
# # # # # # # if cfg.rope_parameters is not None and "rope_theta" in cfg.rope_parameters:
# # # # # # #     rope_theta = cfg.rope_parameters["rope_theta"]

# # # # # # # from theseus.model.models.contrib.qwen import Qwen
# # # # # # # model = Qwen(
# # # # # # #     n_layers=cfg.num_hidden_layers,
# # # # # # #     n_embd=cfg.hidden_size,
# # # # # # #     n_head=cfg.num_attention_heads,
# # # # # # #     n_kv_head=cfg.num_key_value_heads,
# # # # # # #     intermediate_size=cfg.intermediate_size,
# # # # # # #     block_size=cfg.max_position_embeddings,
# # # # # # #     vocab_size=cfg.vocab_size,
# # # # # # #     dropout=0.0,
# # # # # # #     attn_dropout=cfg.attention_dropout,
# # # # # # #     rope_theta=rope_theta,
# # # # # # #     rms_norm_eps=cfg.rms_norm_eps,
# # # # # # #     use_sliding_window=cfg.use_sliding_window,
# # # # # # #     sliding_window=cfg.sliding_window,
# # # # # # #     max_window_layers=cfg.max_window_layers,
# # # # # # #     bias=True,
# # # # # # # )
# # # # # # # dummy = jnp.zeros((1, 1), dtype=jnp.int32)
# # # # # # # shapes = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)

# # # # # # # from theseus.base.job import ExecutionSpec
# # # # # # # spec = ExecutionSpec.local("/sailhome/houjun/theseus")

# # # # # # # import flax

# # # # # # # model_param_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
# # # # # # #     flax.linen.get_partition_spec(shapes),
# # # # # # #     spec.topology.mesh,
# # # # # # #     rules=tuple(model.sharding),
# # # # # # # )

# # # # # # # params = jax.jit(model.init, out_shardings=model_param_sharding)(jax.random.PRNGKey(0), dummy)

# # # # # # # # params = model.init(jax.random.PRNGKey(0), dummy)["params"]
# # # # # # # # praams

# # # # # # # model


# # # # # # # # with quick(PretrainGPT, "test") as j:
# # # # # # # #     j.oco gtgr
# # # # # # # # from theseus.experiments.forking import PretrainThoughtbubbles

# # # # # # # # from theseus.base.job import ExecutionSpec

# # # # # # # # spec = ExecutionSpec.local("/Users/houjun/theseus/")
# # # # # # # # spec
# # # # # # # # block_size = 1024


# # # # # # # # from theseus.experiments.redcodegen import Hardening
# # # # # # # # with quick(Hardening, "test") as j:
# # # # # # # #     j.config.architecture.backbone.implementation = "qwen"
# # # # # # # #     j.config.architecture.backbone.weights = "Qwen/Qwen2.5-0.5B"
# # # # # # # #     j.config.logging.report_interval=1
# # # # # # # #     j.config.architecture.block_size = 1024
# # # # # # # #     j.config.training.per_device_batch_size = 4
# # # # # # # #     j.config.training.batch_size = 32

# # # # # # # #     # contrastive learning, yolo, eventually maybe should
# # # # # # # #     # evaluate i.e. by literally rolling out the model
# # # # # # # #     j.config.training.evaluate = False
# # # # # # # #     j.config.training.validation = False

# # # # # # # #     # tokenizer is qwen
# # # # # # # #     j.config.tokenizer.backend = "huggingingface"
# # # # # # # #     j.config.tokenizer.name = "Qwen/Qwen2.5-0.5B"
# # # # # # # #     j.config.training.dataset = [
# # # # # # # #         {
# # # # # # # #             "name": "redcodegen__hardening",
# # # # # # # #             "suffix": "qwen205b",
# # # # # # # #             "style": "CONTRASTIVE",
# # # # # # # #             "rate": "1.0"
# # # # # # # #         }
# # # # # # # #     ]
# # # # # # #     j.save("./configs/redcodegen/hardeningy.yaml", n_shards=2)
# # # # # # #     # j()

# # # # # # # j.config.training.dataset

# # # # # # #     j.config


# # # # # # #     j.config.architecture.n_head = 16
# # # # # # #     j.config.architecture.max_block_size = 1024
# # # # # # #     j.config.training.per_device_batch_size = 8
# # # # # # #     j.config.logging.checkpoint_interval=10240
# # # # # # #     j.config.logging.validation_interval=2048
# # # # # # #     j.config.eval.evaluations = ["blimp"]
# # # # # # #     j()
# # # # # # #     # j.save("./configs/"
# # # # # # #     # j()

# # # # # # # job = j.create()
# # # # # # # x,y,mask = job.batch()

# # # # # # # job.state_sharding
# # # # # # # mask
# # # # # # # job
# # # # # # # j.config.training.per_device_batch_size = 1
# # # # # # # j.config.training.batch_size = 2
# # # # # # # j.config.logging.report_interval = 2
# # # # # # # j()
# # # # # # # j.save("./configs/continual/abcd.yaml", chip="h200", n_chips=2)

# # # # # # # cfg
# # # # # # # j.config.architecture.huggingface.model = "meta-llama/Llama-3.1-8B-Instruct"
# # # # # # #     j.config.architecture.n_layers = 16
# # # # # # #     j.config.training.dataset = [[{
# # # # # # #         "name": "fineweb",
# # # # # # #         "rate": 1.0,
# # # # # # #         "style": "PMD",
# # # # # # #         "suffix": "",
# # # # # # #     }]]
# # # # # # #     j.config.training.tokens = [1000000000]
# # # # # # #     j.config.eval.evaluations = ["mnli", "qqp", "sst2", "siqa"]

# # # # # # #     j.config.logging.report_interval=1
# # # # # # #     # j.config.logging.validation_interval=4
# # # # # # #     j.config.training.evaluate = False
# # # # # # #     # j.config.training.batch_size = 6
# # # # # # #     j.config.training.per_device_batch_size = 96
# # # # # # #     # trainer = j.create()
# # # # # # #     j()
# # # # # # #     # 1+1
# # # # # # #     # !nvidia-smi

# # # # # # # # # trainer
# # # # # # # # # import flax
# # # # # # # # # print([i.value.sharding for i in trainer.state.params["_params"].values() if isinstance(i, flax.linen.Partitioned)])
# # # # # # # # # [i.value for i in trainer.state.params["_params"].values() if isinstance(i, flax.linen.Partitioned)]
# # # # # # # # from theseus.data.tokenizer import get_tokenizer, TokenizerConfig

# # # # # # # # kl = sum_x(p(x) * (p(x) - q(x)))
# # # # # # # # tk = get_tokenizer(TokenizerConfig(backend="huggingface", name="meta-llama/Llama-3.1-8B"))
# # # # # # # # tk._tokenizer
# # # # # # # # x.max()


# # # # # # # # x,y,pmd = trainer.batch()
# # # # # # # # trainer.forward(trainer.state, trainer.state.params, (x,y,pmd))

# # # # # # # # # shd = flax.linen.logical_to_mesh_sharding(  # type: ignore
# # # # # # # # #     flax.linen.get_partition_spec(trainer.state),
# # # # # # # # #     trainer.mesh,
# # # # # # # # #     rules=tuple(trainer.model.sharding),  # type: ignore
# # # # # # # # # )

# # # # # # # # # !git fetch && git checkout cdb30b47b94a2397044e8eaa6c12416a9825953f
# # # # # # # # # res

# # # # # # # # # #     1+1
# # # # # # # # # # # ls ~/theseus/data/


# # # # # # # # # # with quick("thoughtbubbles/train/pretrain", "test", "/Users/houjun/theseus") as j:
# # # # # # # # # #     cfg = j.config
# # # # # # # # # #     # j.config.data.dataset = "mnli"
# # # # # # # # #     # j.save("./configs/data/chicken.yaml")

# # # # # # # # # block, params = init(
# # # # # # # # #     ForkingAttention,
# # # # # # # # #     cfg,
# # # # # # # # #     x=jnp.ones((7, cfg.architecture.block_size, cfg.architecture.n_embd)),
# # # # # # # # #     cumulative_scores=jnp.ones((7, cfg.architecture.block_size)),
# # # # # # # # #     token_index=jnp.arange(cfg.architecture.block_size)[None,:].repeat(7, axis=0),
# # # # # # # # # )
# # # # # # # # # params

# # # # # # # # # # with configuration(cfg):
# # # # # # # # # #     fb =  configure(ForkingAttention)

# # # # # # # # # # fb.init(jax.random.PRNGKey(7),
