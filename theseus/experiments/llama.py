import optax

from theseus.evaluation.huggingface import HFEvaluator
from theseus.model.models import Llama
from theseus.training.huggingface import HFTrainer, HFTrainerConfig


class EvaluateLlama(HFEvaluator[Llama]):
    MODEL = Llama


class PretrainLlama(HFTrainer[Llama]):
    MODEL = Llama
    CONFIG = HFTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule
