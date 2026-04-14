# Jobs are registered via @job decorators in their definition modules.
try:
    from .abcd import ABCDTrainer, ABCDKLTrainer  # noqa: F401
except ImportError:
    pass

try:
    from .benchmark import (  # noqa: F401
        BenchmarkTransformer,
        BenchmarkMamba,
        BenchmarkHybrid,
        BenchmarkTransformerLoRA,
        BenchmarkMambaLoRA,
        BenchmarkHybridLoRA,
    )
except ImportError:
    pass
