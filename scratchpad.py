# import sys
# from loguru import logger

# logger.remove()
# logger.add(
#     sys.stderr,
#     format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
#     "<level>{level: ^8}</level>| "
#     "<magenta>({name}:{line})</magenta> <level>{message}</level>",
#     level="DEBUG",
#     colorize=True,
#     enqueue=True,
#     filter=lambda x: x["extra"].get("task", "") != "plot",
# )

# from theseus.training.trainers import PretrainGPT
# from theseus.config import *

# cfg = build(*PretrainGPT.config())
# cfg.architecture.n_embd = 128
# cfg.architecture.n_layers = 4

# with configuration(cfg):
#     job = PretrainGPT.local("/sailhome/houjun/theseus", name="test")
