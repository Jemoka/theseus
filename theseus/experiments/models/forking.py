import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation import Evaluator
from theseus.model.models import Thoughtbubbles


class EvaluateThoughtbubbles(Evaluator[Thoughtbubbles]):
    MODEL = Thoughtbubbles


class PretrainThoughtbubbles(BaseTrainer[BaseTrainerConfig, Thoughtbubbles]):
    MODEL = Thoughtbubbles
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule

    def evaluator(self) -> Evaluator[Thoughtbubbles]:
        return EvaluateThoughtbubbles.from_trainer(self)
