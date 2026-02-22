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


# with quick(PretrainGPT, "test") as j:
#     j.oco gtgr
# from theseus.experiments.forking import PretrainThoughtbubbles

# from theseus.base.job import ExecutionSpec

# spec = ExecutionSpec.local("/Users/houjun/theseus/")
# spec
# block_size = 1024


from theseus.experiments.redcodegen import Hardening
with quick(Hardening, "test", "/Users/houjun/theseus") as j:
    ...

#     j.config.architecture.n_head = 16
#     j.config.architecture.max_block_size = 1024
#     j.config.architecture.block_size = 512
#     j.config.training.per_device_batch_size = 8
#     j.config.logging.report_interval=32
#     j.config.logging.checkpoint_interval=10240
#     j.config.logging.validation_interval=2048
#     j.config.eval.evaluations = ["blimp"]
#     j()
#     # j.save("./configs/"
#     # j()

# job = j.create()
# x,y,mask = job.batch()

# job.state_sharding
# mask
# job
# j.config.training.per_device_batch_size = 1
# j.config.training.batch_size = 2
# j.config.logging.report_interval = 2
# j()
# j.save("./configs/continual/abcd.yaml", chip="h200", n_chips=2)

# cfg
# j.config.architecture.huggingface.model = "meta-llama/Llama-3.1-8B-Instruct"
#     j.config.architecture.n_layers = 16
#     j.config.training.dataset = [[{
#         "name": "fineweb",
#         "rate": 1.0,
#         "style": "PMD",
#         "suffix": "",
#     }]]
#     j.config.training.tokens = [1000000000]
#     j.config.eval.evaluations = ["mnli", "qqp", "sst2", "siqa"]

#     j.config.logging.report_interval=1
#     # j.config.logging.validation_interval=4
#     j.config.training.evaluate = False
#     # j.config.training.batch_size = 6
#     j.config.training.per_device_batch_size = 96
#     # trainer = j.create()
#     j()
#     # 1+1
#     # !nvidia-smi

# # # trainer
# # # import flax
# # # print([i.value.sharding for i in trainer.state.params["_params"].values() if isinstance(i, flax.linen.Partitioned)])
# # # [i.value for i in trainer.state.params["_params"].values() if isinstance(i, flax.linen.Partitioned)]
# # from theseus.data.tokenizer import get_tokenizer, TokenizerConfig

# # kl = sum_x(p(x) * (p(x) - q(x)))
# # tk = get_tokenizer(TokenizerConfig(backend="huggingface", name="meta-llama/Llama-3.1-8B"))
# # tk._tokenizer
# # x.max()


# # x,y,pmd = trainer.batch()
# # trainer.forward(trainer.state, trainer.state.params, (x,y,pmd))

# # # shd = flax.linen.logical_to_mesh_sharding(  # type: ignore
# # #     flax.linen.get_partition_spec(trainer.state),
# # #     trainer.mesh,
# # #     rules=tuple(trainer.model.sharding),  # type: ignore
# # # )

# # # !git fetch && git checkout cdb30b47b94a2397044e8eaa6c12416a9825953f
# # # res

# # # #     1+1
# # # # # ls ~/theseus/data/


# # # # with quick("thoughtbubbles/train/pretrain", "test", "/Users/houjun/theseus") as j:
# # # #     cfg = j.config
# # # #     # j.config.data.dataset = "mnli"
# # #     # j.save("./configs/data/chicken.yaml")

# # # block, params = init(
# # #     ForkingAttention,
# # #     cfg,
# # #     x=jnp.ones((7, cfg.architecture.block_size, cfg.architecture.n_embd)),
# # #     cumulative_scores=jnp.ones((7, cfg.architecture.block_size)),
# # #     token_index=jnp.arange(cfg.architecture.block_size)[None,:].repeat(7, axis=0),
# # # )
# # # params

# # # # with configuration(cfg):
# # # #     fb =  configure(ForkingAttention)

# # # # fb.init(jax.random.PRNGKey(7),
