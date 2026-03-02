# Jobs are registered via @job decorators in their definition modules.
try:
    from .abcd import ABCDTrainer, ABCDKLTrainer  # noqa: F401
except ImportError:
    pass
