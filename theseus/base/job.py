"""
Things you can train/infer/etc.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field

from theseus.base.topology import Topology
from theseus.base.hardware import HardwareResult, local


class JobSpec(BaseModel):
    """user provided specification for a job"""

    name: str = Field(description="name of the job, useful for logging, etc.")
    id: Optional[str] = Field(
        description="ID (such as for wandb) of the job, could be None", default=None
    )
    project: Optional[str] = Field(
        description="project this run belongs to", default="general"
    )
    group: Optional[str] = Field(
        description="group under the project this run belongs to", default="default"
    )


class ExecutionSpec(JobSpec):
    """actually allocated specification for a job"""

    topology: Optional[Topology] = Field(description="hardware topology", default=None)
    hardware: HardwareResult = Field(description="actual allocated hardware")
    distributed: bool = Field(
        description="whether or not the run is happening over multiple hosts"
    )

    @classmethod
    def local(
        cls,
        root_dir: str,
        name: str = "local",
        project: str | None = None,
        group: str | None = None,
    ) -> "ExecutionSpec":
        hardware = local(root_dir, "-")
        if hardware.chip is None:
            topology = None
        else:
            topology = Topology.new(hardware.chip)
        spec = cls(
            name=name,
            project=project,
            group=group,
            hardware=hardware,
            distributed=False,
            topology=topology,
        )

        return spec


class _BaseJob(ABC):
    @abstractmethod
    def __call__(self) -> None:
        """We will call this function simultaneously on all hosts"""
        None
