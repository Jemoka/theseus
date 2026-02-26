import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.model.models import Thoughtbubbles


class PretrainThoughtbubbles(BaseTrainer[BaseTrainerConfig, Thoughtbubbles]):
    MODEL = Thoughtbubbles
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule
