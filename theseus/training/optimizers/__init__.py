from .adamw import adamw, AdamWConfig

OPTIMIZERS = {"adamw": (adamw, AdamWConfig)}

__all__ = [
    "OPTIMIZERS",
    "adamw",
    "AdamWConfig",
]
