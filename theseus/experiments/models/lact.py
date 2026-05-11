"""LaCT pretraining job.

Registers ``@job("lact/train/pretrain")`` and the evaluator that pairs with it.
The evaluator inherits from ``TTTInferenceJob`` so its ``forward`` auto-threads
the ``"fast_weights"`` mutable collection during rollouts.
"""

from typing import Any, cast

import optax

from theseus.evaluation.base import Evaluator
from theseus.inference.ttt import TTTInferenceJob
from theseus.model.models import LaCT
from theseus.registry import job
from theseus.training.base import BaseTrainer, BaseTrainerConfig


class TTTEvaluator(TTTInferenceJob, Evaluator[Any]):
    """Evaluator + TTT inference: full evaluator surface, fast-weight mutation.

    Method-resolution order is ``TTTEvaluator → TTTInferenceJob → Evaluator →
    InferenceJob``, so ``forward`` comes from ``TTTInferenceJob`` (mutates
    ``fast_weights``) and the rest of the evaluator behavior (``from_trainer``,
    ``rollout``, ``evaluate``, ``run``) comes from ``Evaluator``.
    """

    pass


@job("lact/train/pretrain")
class PretrainLaCT(BaseTrainer[BaseTrainerConfig, LaCT]):
    MODEL = LaCT
    CONFIG = BaseTrainerConfig

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return "wsd"

    def evaluator(self) -> "TTTEvaluator":
        return cast("TTTEvaluator", TTTEvaluator.from_trainer(self))
