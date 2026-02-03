"""
Quick job runner for rapid prototyping and testing.

Usage:
    from theseus.quick import quick
    from theseus.registry import JOBS

    job = JOBS["continual/train/abcd"]

    with quick(job, "/sailhome/houjun/theseus/", "test") as j:
        j.config.logging.checkpoint_interval = 16384
        j.config.logging.validation_interval = 2048
        j.config.training.per_device_batch_size = 16

        j()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, TYPE_CHECKING

from theseus.config import build, configuration
from theseus.registry import JOBS

if TYPE_CHECKING:
    from omegaconf import DictConfig


class QuickJob:
    """Wrapper for quick job execution."""

    def __init__(
        self,
        job_cls: Any,
        out_path: str,
        name: str,
        project: str | None = None,
        group: str | None = None,
    ):
        self._job_cls = job_cls
        self._out_path = out_path
        self._name = name
        self._project = project
        self._group = group
        self._instance: Any = None

        # Build config from job class
        job_config = job_cls.config()
        if isinstance(job_config, (list, tuple)):
            self.config: DictConfig = build(*job_config)
        else:
            self.config = build(job_config)

    def __call__(self) -> Any:
        """Run the job."""
        self._instance = self._job_cls.local(
            self._out_path,
            name=self._name,
            project=self._project,
            group=self._group,
        )
        return self._instance()


@contextmanager
def quick(
    job: Any | str,
    out_path: str,
    name: str,
    project: str | None = None,
    group: str | None = None,
) -> Generator[QuickJob, None, None]:
    """Context manager for quick job execution.

    Args:
        job: Job class or job name string (e.g., "continual/train/abcd")
        out_path: Output path for job results
        name: Name of the job run
        project: Optional project name
        group: Optional group name

    Yields:
        QuickJob instance with .config attribute for modification

    Example:
        with quick(JOBS["my/job"], "/path/to/output", "test") as j:
            j.config.training.batch_size = 32
            j()
    """
    # Resolve job name to class if string
    if isinstance(job, str):
        if job not in JOBS:
            raise ValueError(
                f"Job '{job}' not found in registry. Available: {list(JOBS.keys())}"
            )
        job = JOBS[job]

    quick_job = QuickJob(job, out_path, name, project, group)

    with configuration(quick_job.config):
        yield quick_job
