import optax

from theseus.model.models import MoEGPT
from theseus.registry import job
from theseus.training.base import BaseTrainer, BaseTrainerConfig


@job("moe/train/pretrain")
class PretrainMoE(BaseTrainer[BaseTrainerConfig, MoEGPT]):
    MODEL = MoEGPT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
