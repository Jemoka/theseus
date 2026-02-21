import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation import Evaluator
from theseus.model.models import GPT


class EvaluateGPT(Evaluator[GPT]):
    MODEL = GPT


class PretrainGPT(BaseTrainer[BaseTrainerConfig, GPT]):
    MODEL = GPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule

    def evaluator(self) -> Evaluator[GPT]:
        return EvaluateGPT.from_trainer(self)
