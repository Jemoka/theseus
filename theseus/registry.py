"""
registry.py
Aggregates jobs, datasets, and evals across the Theseus library for easy access.
"""

from theseus.data import JOBS as DATA_JOBS
from theseus.data.datasets.registry import DATASETS

JOBS = {}
JOBS.update(DATA_JOBS)

__all__ = [
    "JOBS",
    "DATASETS",
]
