"""
registry.py
Decorator-based registry for jobs, datasets, and evaluations.

Decorators (@job, @dataset, @evaluation) can be imported cheaply and used
to register classes at definition time — no heavy submodule imports happen
until ``ensure_registered()`` is called.

Usage:
    from theseus.registry import job, dataset, evaluation

    @job("gpt/train/pretrain")
    class PretrainGPT(BaseTrainer[...]): ...

    @dataset("alpaca")
    class Alpaca(ChatTemplateDataset): ...

    @evaluation("bbq")
    class BBQEval(RolloutEvaluation): ...

User-defined jobs in scripts are recognized automatically — decorate with
@job before calling ``ensure_registered()`` and the class will appear in
JOBS alongside the built-in entries.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

JOBS: dict[str, type] = {}
DATASETS: dict[str, type] = {}
EVALUATIONS: dict[str, Callable[[], Any]] = {}

_registered = False


def ensure_registered() -> None:
    """Trigger registration of all built-in decorated classes.

    Safe to call multiple times — only the first call imports the submodules.
    Any classes already registered via decorators (e.g. user-defined jobs)
    are preserved.
    """
    global _registered
    if _registered:
        return
    _registered = True

    import theseus.data.datasets  # noqa: F401 — dataset decorators
    import theseus.data.tokenize  # noqa: F401 — data job decorators
    import theseus.experiments  # noqa: F401 — experiment job decorators
    import theseus.evaluation.datasets  # noqa: F401 — evaluation decorators


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


__all__ = [
    "JOBS",
    "DATASETS",
    "EVALUATIONS",
    "ensure_registered",
    "job",
    "dataset",
    "evaluation",
]
