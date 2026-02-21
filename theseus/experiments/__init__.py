from typing import Any

from theseus.job import BasicJob
from .models.gpt import PretrainGPT, EvaluateGPT
from .models.forking import PretrainThoughtbubbles, EvaluateThoughtbubbles
from .models.qwen import PretrainQwen, EvaluateQwen
from .continual import JOBS as CONTINUAL_JOBS

JOBS: dict[str, type[BasicJob[Any]]] = {
    "gpt/train/pretrain": PretrainGPT,
    "gpt/eval/evaluate": EvaluateGPT,
    "thoughtbubbles/train/pretrain": PretrainThoughtbubbles,
    "thoughtbubbles/eval/evaluate": EvaluateThoughtbubbles,
    "qwen/train/pretrain": PretrainQwen,
    "qwen/eval/evaluate": EvaluateQwen,
}
JOBS.update(CONTINUAL_JOBS)

__all__ = ["JOBS"]
