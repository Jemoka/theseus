from theseus.base import local, ExecutionSpec
from theseus.data import PreparePretrainingDatasetConfig, PreparePretrainingDatasetJob

hardware = local("/Users/houjun/theseus/theseus-prod-fs", "/Users/houjun/Worktrees")
spec = ExecutionSpec(name="test", hardware=hardware, distributed=False)
config = PreparePretrainingDatasetConfig(name="fineweb")
spec.model_dump()
job = PreparePretrainingDatasetJob(config, spec)
job()
