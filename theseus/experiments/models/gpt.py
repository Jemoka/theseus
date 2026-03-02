import optax

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.model.models import GPT
from theseus.registry import job


@job("gpt/train/pretrain")
class PretrainGPT(BaseTrainer[BaseTrainerConfig, GPT]):
    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule
