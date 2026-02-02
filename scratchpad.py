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

from theseus.experiments.continual.abcd import ABCDConfig, ABCDTrainer, ABCDConfig
from theseus.config import *

build(ABCDConfig)

# class ABCDConfig(BaseTrainerConfig):
#     datasets: str = "hi"

# [(i.name, i.type) for i in fields(ABCDConfig)]



# cfg
# cfg

# cfg.architecture.n_embd = 128
# cfg.architecture.n_layers = 4
# cfg.eval.evaluations = ["longbench"]

# with configuration(cfg):
#     job = PretrainGPT.local("/Users/houjun/theseus/", name="test")
