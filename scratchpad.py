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


from omegaconf import OmegaConf
from theseus.base.hardware import HardwareRequest
from theseus.base.chip import SUPPORTED_CHIPS
from theseus.base.job import JobSpec
from theseus.dispatch import dispatch, load_dispatch_config

# Load dispatch configuration
dispatch_config = load_dispatch_config("/Users/houjun/.theseus.yaml")
dispatch_config

# Define job config (must have 'job' key pointing to registered job)
cfg = OmegaConf.load("/Users/houjun/Downloads/test.yaml")

# Define job specification
spec = JobSpec(
    name="test",
)

# Define hardware requirements
hardware = HardwareRequest(
    chip=SUPPORTED_CHIPS["gb10"],
    min_chips=1,
)

# Dispatch! Returns SlurmResult or RunResult
result = dispatch(
    cfg=cfg,
    spec=spec,
    hardware=hardware,
    dispatch_config=dispatch_config,
    dirty=True,  # include uncommitted changes
)

if result.ok:
    print(f"Job dispatched: {result}")
else:
    print(f"Failed: {result.stderr}")


