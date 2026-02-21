from typing import Any

from theseus.job import BasicJob
from .models.gpt import PretrainGPT, EvaluateGPT
from .models.forking import PretrainThoughtbubbles, EvaluateThoughtbubbles
from .models.qwen import PretrainQwen, EvaluateQwen
from .models.llama import PretrainLlama, EvaluateLlama
from .models.gpt_neox import PretrainGPTNeoX, EvaluateGPTNeoX
from .continual import JOBS as CONTINUAL_JOBS

JOBS: dict[str, type[BasicJob[Any]]] = {
    "gpt/train/pretrain": PretrainGPT,
    "gpt/eval/evaluate": EvaluateGPT,
    "thoughtbubbles/train/pretrain": PretrainThoughtbubbles,
    "thoughtbubbles/eval/evaluate": EvaluateThoughtbubbles,
    "qwen/train/pretrain": PretrainQwen,
    "qwen/eval/evaluate": EvaluateQwen,
    "llama/train/pretrain": PretrainLlama,
    "llama/eval/evaluate": EvaluateLlama,
    "gpt_neox/train/pretrain": PretrainGPTNeoX,
    "gpt_neox/eval/evaluate": EvaluateGPTNeoX,
}
JOBS.update(CONTINUAL_JOBS)

__all__ = ["JOBS"]
