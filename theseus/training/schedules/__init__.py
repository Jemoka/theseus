from .wsd import wsd, WSDConfig
from .wsds import wsds, WSDSConfig

SCHEDULES = {"wsd": (wsd, WSDConfig), "wsds": (wsds, WSDSConfig)}

__all__ = ["SCHEDULES", "wsd", "WSDConfig", "wsds", "WSDSConfig"]
