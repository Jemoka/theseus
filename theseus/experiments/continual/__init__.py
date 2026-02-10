from theseus.job import BasicJob
from typing import Any

from .abcd import ABCDTrainer, ABCDHFTrainer

JOBS: dict[str, type[BasicJob[Any]]] = {
    "continual/train/abcd": ABCDTrainer,
    "continual/train/abcd_hf": ABCDHFTrainer,
}
