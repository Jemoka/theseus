import optax

from theseus.training.trainer import BaseTrainer
from theseus.evaluation import Evaluator
from theseus.model.models import GPT


class PretrainGPT(BaseTrainer[GPT]):
    MODEL = GPT

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule


class EvaluateGPT(Evaluator[GPT]):
    MODEL = GPT
