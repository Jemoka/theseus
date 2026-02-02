import optax

from theseus.training.trainers.base import BaseTrainer
from theseus.model.models import GPT


class PretrainGPT(BaseTrainer[GPT]):
    MODEL = GPT

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule
