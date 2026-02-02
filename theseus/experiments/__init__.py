from typing import Any

from theseus.job import BasicJob
from .gpt import PretrainGPT, EvaluateGPT

JOBS: dict[str, type[BasicJob[Any]]] = {
    "gpt/train/pretrain": PretrainGPT,
    "gpt/eval/evaluate": EvaluateGPT,
}

__all__ = ["JOBS"]
