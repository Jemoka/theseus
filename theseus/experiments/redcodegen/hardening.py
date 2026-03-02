import optax
from theseus.training.contrastive import BackbonedContrastiveTrainer
from theseus.registry import job


@job("redcodegen/train/hardening")
class Hardening(BackbonedContrastiveTrainer):
    """Harden a model by running cybersecurity contrastive datasets via DPO.

    ... this is secretly just standard contrastive learning.
    """

    @classmethod
    def schedule(cls) -> optax._src.base.Schedule:
        return None
