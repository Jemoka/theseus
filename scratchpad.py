from theseus.registry import JOBS
from theseus.config import build, hydrate

tokenize_job = JOBS["data/tokenize_blockwise_dataset"]
config = build(tokenize_job.config)

config.data.dataset = "mnli"

hydrate(tokenize_job.config, config)
