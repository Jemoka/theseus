from typing import Any

from theseus.job import CheckpointedJob

from .pretrainer import (
    Pretrainer,
    PretrainerConfig,
    GPTPretrainer,
    configure_optimizers_adamw,
)

JOBS: dict[str, type[CheckpointedJob[Any]]] = {
    "train/gpt": GPTPretrainer,
}

__all__ = [
    "Pretrainer",
    "PretrainerConfig",
    "GPTPretrainer",
    "configure_optimizers_adamw",
    "JOBS",
]
