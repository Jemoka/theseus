"""
registry.py
Aggregates jobs, datasets, and evals across the Theseus library for easy access.
"""

from typing import Any

from theseus.data import JOBS as DATA_JOBS
from theseus.data.datasets.registry import DATASETS
from theseus.job import BasicJob

JOBS: dict[str, type[BasicJob[Any]]] = {}
JOBS.update(DATA_JOBS)

__all__ = [
    "JOBS",
    "DATASETS",
]
