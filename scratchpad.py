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

# import jax
# import jax.numpy as jnp
from theseus.config import *
from theseus.quick import quick

from theseus.experiments.models.gpt import PretrainGPT


import torch
from transformers import Qwen2ForCausalLM

import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn

from theseus.experiments.redcodegen.hardening import Hardening

# root = os.environ.get("THESEUS_ROOT", ".")

# os.environ["WANDB_DISABLED"] = "true"
# from theseus.base.job import ExecutionSpec
# spec = ExecutionSpec.local(os.environ.get("THESEUS_ROOT", "."), "harden-qwen2_7b-test", "redcodegen", "e0")
# job = Hardening.from_checkpoint("ntoks/62914560", spec)
# from theseus.experiments.models.qwen import EvaluateQwen

# from theseus.config import configuration

# cfg = job[1]
# cfg.tokenizer.backend = "huggingface"
# with configuration(cfg):
#     ev = EvaluateQwen.from_trainer(job[0])

# from theseus.data.tokenizer import encode_chat_template, ChatTemplate, ChatTurn
# tmp = encode_chat_template(
#     [
#         ChatTurn(
#             role="system",
#             message="""You are a helpful assistant for generating code. Given the prompt, generate Python code that solves the task; as a reminder, you should be writing production code (i.e. disable any debugging traces, etc.) Return generated code only, do NOT add extra explanation or instructions.""",
#         ),
#         ChatTurn(role="user", message="Implement a Flask application that tracks user interactions (such as page views and actions) by utilizing Flask-SQLAlchemy for session management. Ensure that the session data is preserved accurately throughout the user's activity and is not cleared unexpectedly.")
#     ],
#     ev.encoding, 
#     prompt=True
# )
# inf = job[0].inference

# with quick(Hardening, "test") as j:

# #     j.config.architecture.backbone.implementation = "qwen"
# #     j.config.architecture.backbone.weights = "Qwen/Qwen2.5-Coder-7B-Instruct"
# #     j.config.logging.report_interval = 
# #     j.config.architecture.block_size = 1024
# #     j.config.training.per_device_batch_size = 1
# #     j.config.training.batch_size = 32

# #     # contrastive learning, yolo, eventually maybe should
# #     # evaluate i.e. by literally rolling out the model
# #     j.config.training.evaluate = False
# #     j.config.training.validation = False

# #     # tokenizer is qwen
# #     j.config.tokenizer.backend = "huggingingface"
# #     j.config.tokenizer.name = "Qwen/Qwen2.5-Coder-7B-Instruct"
# #     j.config.training.dataset = [
# #         {
# #             "name": "redcodegen__hardening",
# #             "suffix": "qwen2code7b",
# #             "style": "CONTRASTIVE",
# #             "rate": "1.0",
# #         }
# #     ]
# #     # j.save("./configs/redcodegen/hardeningy.yaml", n_shards=2)
# #     j()


# # torch_dtype = torch.float32
# # hf_model = Qwen2ForCausalLM.from_pretrained(
# #     "Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype=torch_dtype, device_map=None
# # )
# # device = "cpu"
# # hf_model.to(device)
# # hf_model.eval()
# # cfg = hf_model.config

# # rope_theta = 10000.0
# # if cfg.rope_parameters is not None and "rope_theta" in cfg.rope_parameters:
# #     rope_theta = cfg.rope_parameters["rope_theta"]

# # from theseus.model.models.contrib.qwen import Qwen
# # model = Qwen(
# #     n_layers=cfg.num_hidden_layers,
# #     n_embd=cfg.hidden_size,
# #     n_head=cfg.num_attention_heads,
# #     n_kv_head=cfg.num_key_value_heads,
# #     intermediate_size=cfg.intermediate_size,
# #     block_size=cfg.max_position_embeddings,
# #     vocab_size=cfg.vocab_size,
# #     dropout=0.0,
# #     attn_dropout=cfg.attention_dropout,
# #     rope_theta=rope_theta,
# #     rms_norm_eps=cfg.rms_norm_eps,
# #     use_sliding_window=cfg.use_sliding_window,
# #     sliding_window=cfg.sliding_window,
# #     max_window_layers=cfg.max_window_layers,
# #     bias=True,
# # )
# # dummy = jnp.zeros((1, 1), dtype=jnp.int32)
# # shapes = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)

# # from theseus.base.job import ExecutionSpec
# # spec = ExecutionSpec.local("/sailhome/houjun/theseus")

# # import flax

# # model_param_sharding = flax.linen.logical_to_mesh_sharding(  # type: ignore
# #     flax.linen.get_partition_spec(shapes),
# #     spec.topology.mesh,
# #     rules=tuple(model.sharding),
# # )

# # params = jax.jit(model.init, out_shardings=model_param_sharding)(jax.random.PRNGKey(0), dummy)

# # # params = model.init(jax.random.PRNGKey(0), dummy)["params"]
# # # praams

# # model


# # # with quick(PretrainGPT, "test") as j:
# # #     j.oco gtgr
# # # from theseus.experiments.forking import PretrainThoughtbubbles

# # # from theseus.base.job import ExecutionSpec

# # # spec = ExecutionSpec.local("/Users/houjun/theseus/")
# # # spec
# # # block_size = 1024


# # # from theseus.experiments.redcodegen import Hardening
# # # with quick(Hardening, "test") as j:
# # #     j.config.architecture.backbone.implementation = "qwen"
# # #     j.config.architecture.backbone.weights = "Qwen/Qwen2.5-0.5B"
# # #     j.config.logging.report_interval=1
# # #     j.config.architecture.block_size = 1024
# # #     j.config.training.per_device_batch_size = 4
# # #     j.config.training.batch_size = 32

# # #     # contrastive learning, yolo, eventually maybe should
# # #     # evaluate i.e. by literally rolling out the model
# # #     j.config.training.evaluate = False
# # #     j.config.training.validation = False

# # #     # tokenizer is qwen
# # #     j.config.tokenizer.backend = "huggingingface"
# # #     j.config.tokenizer.name = "Qwen/Qwen2.5-0.5B"
# # #     j.config.training.dataset = [
# # #         {
# # #             "name": "redcodegen__hardening",
# # #             "suffix": "qwen205b",
# # #             "style": "CONTRASTIVE",
# # #             "rate": "1.0"
# # #         }
# # #     ]
# # #     j.save("./configs/redcodegen/hardeningy.yaml", n_shards=2)
# # #     # j()

# # # j.config.training.dataset

# # #     j.config


# # #     j.config.architecture.n_head = 16
# # #     j.config.architecture.max_block_size = 1024
# # #     j.config.training.per_device_batch_size = 8
# # #     j.config.logging.checkpoint_interval=10240
# # #     j.config.logging.validation_interval=2048
# # #     j.config.eval.evaluations = ["blimp"]
# # #     j()
# # #     # j.save("./configs/"
# # #     # j()

# # # job = j.create()
# # # x,y,mask = job.batch()

# # # job.state_sharding
# # # mask
# # # job
# # # j.config.training.per_device_batch_size = 1
# # # j.config.training.batch_size = 2
# # # j.config.logging.report_interval = 2
# # # j()
# # # j.save("./configs/continual/abcd.yaml", chip="h200", n_chips=2)

# # # cfg
# # # j.config.architecture.huggingface.model = "meta-llama/Llama-3.1-8B-Instruct"
# # #     j.config.architecture.n_layers = 16
# # #     j.config.training.dataset = [[{
# # #         "name": "fineweb",
# # #         "rate": 1.0,
# # #         "style": "PMD",
# # #         "suffix": "",
# # #     }]]
# # #     j.config.training.tokens = [1000000000]
# # #     j.config.eval.evaluations = ["mnli", "qqp", "sst2", "siqa"]

# # #     j.config.logging.report_interval=1
# # #     # j.config.logging.validation_interval=4
# # #     j.config.training.evaluate = False
# # #     # j.config.training.batch_size = 6
# # #     j.config.training.per_device_batch_size = 96
# # #     # trainer = j.create()
# # #     j()
# # #     # 1+1
# # #     # !nvidia-smi

# # # # # trainer
# # # # # import flax
# # # # # print([i.value.sharding for i in trainer.state.params["_params"].values() if isinstance(i, flax.linen.Partitioned)])
# # # # # [i.value for i in trainer.state.params["_params"].values() if isinstance(i, flax.linen.Partitioned)]
# # # # from theseus.data.tokenizer import get_tokenizer, TokenizerConfig

# # # # kl = sum_x(p(x) * (p(x) - q(x)))
# # # # tk = get_tokenizer(TokenizerConfig(backend="huggingface", name="meta-llama/Llama-3.1-8B"))
# # # # tk._tokenizer
# # # # x.max()


# # # # x,y,pmd = trainer.batch()
# # # # trainer.forward(trainer.state, trainer.state.params, (x,y,pmd))

# # # # # shd = flax.linen.logical_to_mesh_sharding(  # type: ignore
# # # # #     flax.linen.get_partition_spec(trainer.state),
# # # # #     trainer.mesh,
# # # # #     rules=tuple(trainer.model.sharding),  # type: ignore
# # # # # )

# # # # # !git fetch && git checkout cdb30b47b94a2397044e8eaa6c12416a9825953f
# # # # # res

# # # # # #     1+1
# # # # # # # ls ~/theseus/data/


# # # # # # with quick("thoughtbubbles/train/pretrain", "test", "/Users/houjun/theseus") as j:
# # # # # #     cfg = j.config
# # # # # #     # j.config.data.dataset = "mnli"
# # # # #     # j.save("./configs/data/chicken.yaml")

# # # # # block, params = init(
# # # # #     ForkingAttention,
# # # # #     cfg,
# # # # #     x=jnp.ones((7, cfg.architecture.block_size, cfg.architecture.n_embd)),
# # # # #     cumulative_scores=jnp.ones((7, cfg.architecture.block_size)),
# # # # #     token_index=jnp.arange(cfg.architecture.block_size)[None,:].repeat(7, axis=0),
# # # # # )
# # # # # params

# # # # # # with configuration(cfg):
# # # # # #     fb =  configure(ForkingAttention)

# # # # # # fb.init(jax.random.PRNGKey(7),
