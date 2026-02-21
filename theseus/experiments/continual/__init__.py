from theseus.job import BasicJob
from typing import Any

from .abcd import ABCDTrainer

JOBS: dict[str, type[BasicJob[Any]]] = {"continual/train/abcd": ABCDTrainer}
