import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models import Qwen


class PretrainQwen(BaseTrainer[BaseTrainerConfig, Qwen]):
    MODEL = Qwen
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule


class FinetuneBackboneQwen(BackbonedTrainer):
    """Finetune from a pretrained Qwen backbone.

    Config keys:
        architecture/backbone/implementation: "qwen"
        architecture/backbone/weights: e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
