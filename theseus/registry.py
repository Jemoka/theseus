"""
registry.py
Decorator-based registry for jobs, datasets, and evaluations.

Usage:
    from theseus.registry import job, dataset, evaluation

    @job("gpt/train/pretrain")
    class PretrainGPT(BaseTrainer[...]): ...

    @dataset("alpaca")
    class Alpaca(ChatTemplateDataset): ...

    @evaluation("bbq")
    class BBQEval(RolloutEvaluation): ...
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

JOBS: dict[str, type] = {}
DATASETS: dict[str, type] = {}
EVALUATIONS: dict[str, Callable[[], Any]] = {}


def job(key: str) -> Callable[[T], T]:
    """Register a job class under the given key."""

    def decorator(cls: T) -> T:
        JOBS[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def dataset(key: str) -> Callable[[T], T]:
    """Register a dataset class under the given key."""

    def decorator(cls: T) -> T:
        DATASETS[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


def evaluation(key: str) -> Callable[[T], T]:
    """Register an evaluation callable under the given key."""

    def decorator(cls: T) -> T:
        EVALUATIONS[key] = cls  # type: ignore[assignment]
        return cls

    return decorator


# Re-export unchanged registries
from theseus.training.optimizers import OPTIMIZERS  # noqa: E402
from theseus.training.schedules import SCHEDULES  # noqa: E402

# Trigger registration of all decorated classes.
# These imports cause sub-modules to execute, which runs the @job/@dataset/@evaluation
# decorators and populates the dicts above.
import theseus.data.datasets  # noqa: F401, E402 — dataset decorators
import theseus.data.tokenize  # noqa: F401, E402 — data job decorators
import theseus.experiments  # noqa: F401, E402 — experiment job decorators
import theseus.evaluation.datasets  # noqa: F401, E402 — evaluation decorators

__all__ = [
    "JOBS",
    "DATASETS",
    "EVALUATIONS",
    "OPTIMIZERS",
    "SCHEDULES",
    "job",
    "dataset",
    "evaluation",
]
