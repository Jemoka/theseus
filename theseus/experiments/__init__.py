from .gpt import PretrainGPT, EvaluateGPT

JOBS = {
    "gpt/train/pretrain": PretrainGPT,
    "gpt/eval/evaluate": EvaluateGPT,
}

__all__ = ["JOBS"]
