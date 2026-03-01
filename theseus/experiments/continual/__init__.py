from theseus.job import BasicJob
from typing import Any

try:
    from .abcd import ABCDTrainer, ABCDKLTrainer

    JOBS: dict[str, type[BasicJob[Any]]] = {
        "continual/train/abcd": ABCDTrainer,
        "continual/train/abcd_kl": ABCDKLTrainer,
    }
except ImportError:
    JOBS = {}
