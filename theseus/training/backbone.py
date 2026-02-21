"""
BackbonedTrainer: finetune from a pretrained HuggingFace backbone.

Instead of configuring model architecture from scratch, reads two config keys:
  - architecture/backbone/implementation: "llama", "qwen", or "gpt_neox"
  - architecture/backbone/weights: HuggingFace model ID (e.g. "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

The model class and initial weights are loaded via from_pretrained,
bypassing the normal configure() path for architecture parameters.
"""

from dataclasses import dataclass
from typing import Any, List, Type

from jax import random as jax_random

from theseus.config import field
from theseus.model.module import Module
from theseus.model.models.llama import Llama
from theseus.model.models.qwen import Qwen
from theseus.model.models.gpt_neox import GPTNeoX
from theseus.training.trainer import BaseTrainer, BaseTrainerConfig
from theseus.evaluation.base import EvaluatorConfig
from theseus.data.tokenizer import TokenizerConfig

BACKBONES: dict[str, Any] = {
    "llama": Llama,
    "qwen": Qwen,
    "gpt_neox": GPTNeoX,
}


@dataclass
class BackboneConfig:
    implementation: str = field("architecture/backbone/implementation")
    weights: str = field("architecture/backbone/weights")


class BackbonedTrainer(BaseTrainer[BaseTrainerConfig, Module]):
    """Trainer that initializes from a pretrained HuggingFace backbone."""

    MODEL: Type[Any] = Module
    CONFIG = BaseTrainerConfig

    @classmethod
    def _config(cls) -> List[Type[Any]]:
        # No cls.MODEL.gather() — architecture comes from HF
        from theseus.training.optimizers import OPTIMIZERS
        from theseus.training.schedules import SCHEDULES

        cfg: List[Type[Any]] = [BackboneConfig, EvaluatorConfig, TokenizerConfig]

        optim = cls.optimizer()
        if isinstance(optim, str):
            result = OPTIMIZERS.get(optim)
            if result is not None:
                cfg.append(result[1])

        sched = cls.schedule()
        if isinstance(sched, str):
            sched_result: Any = SCHEDULES.get(sched)
            if sched_result is not None:
                cfg.append(sched_result[1])

        return cfg

    def _init_model(self) -> Any:
        from theseus.config import configure

        backbone_cfg = configure(BackboneConfig)

        model_cls = BACKBONES[backbone_cfg.implementation]
        self.model, params = model_cls.from_pretrained(backbone_cfg.weights)

        # Still need to split keys for dropout
        self.key, _init_key, self.dropout_key = jax_random.split(self.key, num=3)

        return params

    def evaluator(self) -> Any:
        raise NotImplementedError(
            "BackbonedTrainer subclasses must implement evaluator()"
        )
