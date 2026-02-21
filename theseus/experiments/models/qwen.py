import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation import Evaluator
from theseus.model.models import Qwen


class EvaluateQwen(Evaluator[Qwen]):
    MODEL = Qwen


class PretrainQwen(BaseTrainer[BaseTrainerConfig, Qwen]):
    MODEL = Qwen
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule

    def evaluator(self) -> Evaluator[Qwen]:
        return EvaluateQwen.from_trainer(self)
