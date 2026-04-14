from dataclasses import dataclass
from theseus.config import field

import optax


@dataclass
class CosineRewarmConfig:
    lr: float = field("optimization/lr", default=3e-4)
    warmup_pct: float = field("optimization/warmup_pct", default=0.005)


def cosine_rewarm(
    total_steps: int, cfg: CosineRewarmConfig
) -> optax._src.base.Schedule:
    warmup_steps = int(total_steps * cfg.warmup_pct)
    cosine_steps = total_steps - warmup_steps

    warmup_schedule = optax.linear_schedule(
        init_value=cfg.lr * 0.01, end_value=cfg.lr, transition_steps=warmup_steps
    )
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=cfg.lr, alpha=0.01, decay_steps=cosine_steps
    )

    return optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps],
    )
