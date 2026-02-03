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


with quick("gpt/train/pretrain", "~/theseus/", "test") as j:
    j.config.logging.checkpoint_interval = 16384
    j.config.logging.validation_interval = 2048
    j.config.training.per_device_batch_size = 8
    j.config.eval.evaluations = []

    j.save("./configs/gpt.yaml", "h200", 2)

