"""
registry.py
Decorator-based registry for jobs, datasets, and evaluations.

Decorators (@job, @dataset, @evaluation) can be imported cheaply and used
to register classes at definition time — no heavy submodule imports happen
until the registry dicts are actually read.

Usage:
    from theseus.registry import job, dataset, evaluation

    @job("gpt/train/pretrain")
    class PretrainGPT(BaseTrainer[...]): ...

    @dataset("alpaca")
    class Alpaca(ChatTemplateDataset): ...

    @evaluation("bbq")
    class BBQEval(RolloutEvaluation): ...

User-defined jobs in scripts are recognized automatically — decorate with
@job/@dataset/@evaluation and the class will coexist with built-in entries
the moment any code reads from the registry.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

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


class _LazyRegistry(dict[str, Any]):
    """A dict that calls ensure_registered() on any read access."""

    # --- writes are always direct (decorators must not trigger registration) ---

    # --- reads trigger registration first ---

    def __getitem__(self, key: str) -> Any:
        ensure_registered()
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        ensure_registered()
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        ensure_registered()
        return super().__contains__(key)

    def __iter__(self) -> Iterator[str]:
        ensure_registered()
        return super().__iter__()

    def __len__(self) -> int:
        ensure_registered()
        return super().__len__()

    def keys(self) -> Any:
        ensure_registered()
        return super().keys()

    def values(self) -> Any:
        ensure_registered()
        return super().values()

    def items(self) -> Any:
        ensure_registered()
        return super().items()

    def __repr__(self) -> str:
        ensure_registered()
        return super().__repr__()


JOBS: dict[str, type] = _LazyRegistry()
DATASETS: dict[str, type] = _LazyRegistry()
EVALUATIONS: dict[str, Callable[[], Any]] = _LazyRegistry()


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
