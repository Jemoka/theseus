import optax

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models import Llama


class PretrainLlama(BaseTrainer[BaseTrainerConfig, Llama]):
    MODEL = Llama
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule


class FinetuneBackboneLlama(BackbonedTrainer):
    """Finetune from a pretrained Llama backbone.

    Config keys:
        architecture/backbone/implementation: "llama"
        architecture/backbone/weights: e.g. "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
