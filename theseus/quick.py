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

    # Or save config to file for later submission:
    with quick(job, "/sailhome/houjun/theseus/", "test") as j:
        j.config.training.per_device_batch_size = 16
        j.save("config.yaml", chip="h100", n_chips=8)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, TYPE_CHECKING

from omegaconf import OmegaConf

from theseus.config import build, configuration
from theseus.registry import JOBS

if TYPE_CHECKING:
    from omegaconf import DictConfig


class QuickJob:
    """Wrapper for quick job execution."""

    def __init__(
        self,
        job_cls: Any,
        job_name: str,
        out_path: str,
        name: str,
        project: str | None = None,
        group: str | None = None,
    ):
        self._job_cls = job_cls
        self._job_name = job_name
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

    def save(
        self,
        out_yaml: str,
        chip: str | None = None,
        n_chips: int | None = None,
    ) -> Path:
        """Save the config to a YAML file.

        Args:
            out_yaml: Output path for the YAML config
            chip: Optional chip type for hardware request (e.g., "h100", "a100")
            n_chips: Optional minimum number of chips for hardware request

        Returns:
            Path to the saved config file
        """
        from theseus.base.chip import SUPPORTED_CHIPS

        # Validate chip if specified
        if chip and chip not in SUPPORTED_CHIPS:
            raise ValueError(
                f"Unknown chip '{chip}'. Available: {list(SUPPORTED_CHIPS.keys())}"
            )

        # Make a copy of config to modify
        config = OmegaConf.create(OmegaConf.to_container(self.config))
        OmegaConf.set_struct(config, False)

        # Add job name
        config.job = self._job_name

        # Add hardware request if specified
        if chip or n_chips:
            config.request = OmegaConf.create({})
            if chip:
                config.request.chip = chip
            if n_chips:
                config.request.min_chips = n_chips

        OmegaConf.set_struct(config, True)

        # Write the config
        out_path = Path(out_yaml)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = OmegaConf.to_yaml(config)
        out_path.write_text(yaml_str)

        return out_path


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
        job_name = job
        if job not in JOBS:
            raise ValueError(
                f"Job '{job}' not found in registry. Available: {list(JOBS.keys())}"
            )
        job_cls = JOBS[job]
    else:
        # Try to find the job name from registry
        job_cls = job
        job_name = next(
            (name for name, cls in JOBS.items() if cls is job_cls),
            "unknown",
        )

    quick_job = QuickJob(job_cls, job_name, out_path, name, project, group)

    with configuration(quick_job.config):
        yield quick_job
