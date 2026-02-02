from dataclasses import dataclass
from theseus.config import field

import optax


@dataclass
class AdamWConfig:
    weight_decay: float = field("optimization/weight_decay", default=0.1)
    beta1: float = field("optimization/beta1", default=0.9)
    beta2: float = field("optimization/beta2", default=0.95)


def adamw(
    lr: optax.base.Schedule | float, cfg: AdamWConfig
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr,
            b1=cfg.beta1,
            b2=cfg.beta2,
            weight_decay=cfg.weight_decay,
        ),
    )
