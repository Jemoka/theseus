from typing import Any

from theseus.job import BasicJob
from .gpt import PretrainGPT, EvaluateGPT
from .continual import JOBS as CONTINUAL_JOBS

JOBS: dict[str, type[BasicJob[Any]]] = {
    "gpt/train/pretrain": PretrainGPT,
    "gpt/eval/evaluate": EvaluateGPT,
}
JOBS.update(CONTINUAL_JOBS)

__all__ = ["JOBS"]
