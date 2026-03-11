import optax

from theseus.training.base import BaseTrainerConfig
from theseus.training.sidechannel import (
    SideChannelTrainer,
    SideChannelTrainerConfig,
    SideChannelBackboneTrainer,
)
from theseus.model.models.sidechannel import SideChannelGPT
from theseus.registry import job


@job("sidechannel_gpt/train/pretrain")
class PretrainSideChannelGPT(SideChannelTrainer):
    """Stage 1: Pretrain SideChannelGPT from scratch.

    Cross-attention gates init at 0, so model behaves as vanilla GPT.
    Can use standard data (FineWeb) or side-channel format.
    """

    MODEL = SideChannelGPT
    CONFIG = SideChannelTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"


@job("sidechannel_gpt/train/finetune")
class FinetuneSideChannelGPT(SideChannelTrainer):
    """Stage 2: Finetune SideChannelGPT on side-channel format data (e.g. WildChat)."""

    MODEL = SideChannelGPT
    CONFIG = SideChannelTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"


@job("sidechannel_qwen/train/finetune")
class FinetuneSideChannelQwen(SideChannelBackboneTrainer):
    """Finetune SideChannelQwen from pretrained Qwen 2.5 weights.

    Cross-attention layers init with gates=0, preserving base behavior.
    Inherits from SideChannelBackboneTrainer (extends BackbonedTrainer).

    Config keys:
        architecture/backbone/implementation: "sidechannel_qwen"
        architecture/backbone/weights: "Qwen/Qwen2.5-0.5B-Instruct"
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"
