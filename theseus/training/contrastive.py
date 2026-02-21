from typing import Any

from theseus.training.trainer import BaseTrainer, BaseTrainerConfig, M


class ContrastiveTrainer(BaseTrainer[BaseTrainerConfig, M]):
    """Contrastive trainer scaffold (loss/forward to be filled in)."""

    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        # TODO: implement contrastive forward/loss
        raise NotImplementedError("Contrastive forward not implemented yet.")
