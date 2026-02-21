import optax

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
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


class FinetuneBackboneGPTNeoX(BackbonedTrainer):
    """Finetune from a pretrained GPT-NeoX/Pythia backbone.

    Config keys:
        architecture/backbone/implementation: "gpt_neox"
        architecture/backbone/weights: e.g. "EleutherAI/pythia-70m-deduped"
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"

    def evaluator(self) -> Evaluator[GPTNeoX]:  # type: ignore[override]
        return EvaluateGPTNeoX.from_trainer(self)
