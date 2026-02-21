import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
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


class FinetuneBackboneLlama(BackbonedTrainer):
    """Finetune from a pretrained Llama backbone.

    Config keys:
        architecture/backbone/implementation: "llama"
        architecture/backbone/weights: e.g. "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"

    def evaluator(self) -> Evaluator[Llama]:  # type: ignore[override]
        return EvaluateLlama.from_trainer(self)
