"""
registry.py
Aggregates jobs, datasets, and evals across the Theseus library for easy access.
"""

from typing import Any

from theseus.data import JOBS as DATA_JOBS
from theseus.data.datasets.registry import DATASETS

from theseus.experiments import JOBS as EXPERIMENT_JOBS
from theseus.job import BasicJob
from theseus.training.optimizers import OPTIMIZERS
from theseus.training.schedules import SCHEDULES

JOBS: dict[str, type[BasicJob[Any]]] = {}
JOBS.update(DATA_JOBS)
JOBS.update(EXPERIMENT_JOBS)

__all__ = ["JOBS", "DATASETS", "OPTIMIZERS", "SCHEDULES"]
