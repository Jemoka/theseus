import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation import Evaluator
from theseus.model.models import Llama


class EvaluateLlama(Evaluator[Llama]):
    MODEL = Llama


class PretrainLlama(BaseTrainer[BaseTrainerConfig, Llama]):
    MODEL = Llama
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule

    def evaluator(self) -> Evaluator[Llama]:
        return EvaluateLlama.from_trainer(self)
