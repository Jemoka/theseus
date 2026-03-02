import optax

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.model.models import Thoughtbubbles
from theseus.registry import job


@job("thoughtbubbles/train/pretrain")
class PretrainThoughtbubbles(BaseTrainer[BaseTrainerConfig, Thoughtbubbles]):
    MODEL = Thoughtbubbles
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule
