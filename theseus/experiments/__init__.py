from typing import Any

from theseus.job import BasicJob
from .models.gpt import PretrainGPT
from .models.forking import PretrainThoughtbubbles
from .models.qwen import PretrainQwen, FinetuneBackboneQwen
from .models.llama import PretrainLlama, FinetuneBackboneLlama
from .models.gpt_neox import PretrainGPTNeoX, FinetuneBackboneGPTNeoX

from .continual import JOBS as CONTINUAL_JOBS
from .redcodegen import JOBS as RCG_JOBS

JOBS: dict[str, type[BasicJob[Any]]] = {
    "gpt/train/pretrain": PretrainGPT,
    "thoughtbubbles/train/pretrain": PretrainThoughtbubbles,
    "qwen/train/pretrain": PretrainQwen,
    "llama/train/pretrain": PretrainLlama,
    "gpt_neox/train/pretrain": PretrainGPTNeoX,
    "llama/train/finetune": FinetuneBackboneLlama,
    "qwen/train/finetune": FinetuneBackboneQwen,
    "gpt_neox/train/finetune": FinetuneBackboneGPTNeoX,
}
JOBS.update(CONTINUAL_JOBS)
JOBS.update(RCG_JOBS)

__all__ = ["JOBS"]
