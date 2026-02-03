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


from theseus.quick import quick

with quick("continual/train/abcd", "/sailhome/houjun/theseus", "test") as j:
    j.config.logging.checkpoint_interval = 4096
    j.config.logging.validation_interval = 1024
    j.config.training.per_device_batch_size = 4
    j.config.architecture.n_layers = 16
    j.config.logging.report_interval


    j.save("./configs/continual/abcd.yaml", "a6000", 4)

