from .base import BaseTrainer, BaseTrainerConfig
from .basic.pretrain import PretrainGPT

JOBS = {"train/basic/pretrain_gpt": PretrainGPT}

__all__ = [
    "JOBS",
    "BaseTrainer",
    "BaseTrainerConfig",
    "PretrainGPT",
]
