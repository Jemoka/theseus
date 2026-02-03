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
from theseus.data.tokenize import TokenizeVariableDatasetJob

with quick(TokenizeVariableDatasetJob, "/Users/houjun/theseus", "test") as j:
    j.config.data.dataset = "mnli"
    j.save("./configs/data/chicken.yaml")



