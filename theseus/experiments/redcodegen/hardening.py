import optax
from typing import Optional, Any
from theseus.training.contrastive import BackbonedContrastiveTrainer


class Hardening(BackbonedContrastiveTrainer):
    """Harden a model by running cybersecurity contrastive datasets via DPO.

    ... this is secretly just standard contrastive learning.
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return None

    def evaluator(self) -> Optional[Any]:
        return None
