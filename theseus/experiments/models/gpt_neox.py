import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models import GPTNeoX


class PretrainGPTNeoX(BaseTrainer[BaseTrainerConfig, GPTNeoX]):
    MODEL = GPTNeoX
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"  # warmup-stable-decay schedule


class FinetuneBackboneGPTNeoX(BackbonedTrainer):
    """Finetune from a pretrained GPT-NeoX/Pythia backbone.

    Config keys:
        architecture/backbone/implementation: "gpt_neox"
        architecture/backbone/weights: e.g. "EleutherAI/pythia-70m-deduped"
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
