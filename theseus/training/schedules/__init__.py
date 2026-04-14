from .cosine_rewarm import cosine_rewarm, CosineRewarmConfig
from .wsd import wsd, WSDConfig
from .wsds import wsds, WSDSConfig

SCHEDULES = {
    "wsd": (wsd, WSDConfig),
    "wsds": (wsds, WSDSConfig),
    "cosine_rewarm": (cosine_rewarm, CosineRewarmConfig),
}

__all__ = [
    "SCHEDULES",
    "cosine_rewarm",
    "CosineRewarmConfig",
    "wsd",
    "WSDConfig",
    "wsds",
    "WSDSConfig",
]
