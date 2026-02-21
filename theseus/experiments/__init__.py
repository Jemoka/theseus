from typing import Any

from theseus.job import BasicJob
from .models.gpt import PretrainGPT, EvaluateGPT
from .models.forking import PretrainThoughtbubbles, EvaluateThoughtbubbles
from .models.qwen import PretrainQwen, EvaluateQwen, FinetuneBackboneQwen
from .models.llama import PretrainLlama, EvaluateLlama, FinetuneBackboneLlama
from .models.gpt_neox import PretrainGPTNeoX, EvaluateGPTNeoX, FinetuneBackboneGPTNeoX
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
    "llama/train/finetune": FinetuneBackboneLlama,
    "qwen/train/finetune": FinetuneBackboneQwen,
    "gpt_neox/train/finetune": FinetuneBackboneGPTNeoX,
}
JOBS.update(CONTINUAL_JOBS)

__all__ = ["JOBS"]
