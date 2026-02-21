import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation import Evaluator
from theseus.model.models import GPTNeoX


class EvaluateGPTNeoX(Evaluator[GPTNeoX]):
    MODEL = GPTNeoX


class PretrainGPTNeoX(BaseTrainer[BaseTrainerConfig, GPTNeoX]):
    MODEL = GPTNeoX
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule

    def evaluator(self) -> Evaluator[GPTNeoX]:
        return EvaluateGPTNeoX.from_trainer(self)
