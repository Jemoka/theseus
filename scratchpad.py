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


import jax
import jax.numpy as jnp
from theseus.config import *
from theseus.quick import quick, init
from theseus.data.tokenize import TokenizeVariableDatasetJob

with quick("thoughtbubbles/train/pretrain", "test", "/Users/houjun/theseus") as j:
    cfg = j.config
    # j.config.data.dataset = "mnli"
    # j.save("./configs/data/chicken.yaml")

block, params = init(
    ForkingAttention,
    cfg,
    x=jnp.ones((7, cfg.architecture.block_size, cfg.architecture.n_embd)),
    cumulative_scores=jnp.ones((7, cfg.architecture.block_size)),
    token_index=jnp.arange(cfg.architecture.block_size)[None,:].repeat(7, axis=0),
)
params

# with configuration(cfg):
#     fb =  configure(ForkingAttention)

# fb.init(jax.random.PRNGKey(7), 


