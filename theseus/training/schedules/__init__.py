from .wsd import wsd, WSDConfig

SCHEDULES = {"wsd": (wsd, WSDConfig)}

__all__ = [
    "SCHEDULES",
    "adamw",
    "AdamWConfig",
]
