"""
Things you can train/infer/etc.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field

from theseus.base.topology import Topology
from theseus.base.hardware import HardwareResult


class JobSpec(BaseModel):
    """user provided specification for a job"""

    name: str = Field(description="name of the job, useful for logging, etc.")
    id: Optional[str] = Field(
        description="ID (such as for wandb) of the job, could be None", default=None
    )
    project: Optional[str] = Field(
        description="project this run belongs to", default="theseus"
    )
    group: Optional[str] = Field(
        description="group under the project this run belongs to", default=""
    )


class ExecutionSpec(JobSpec):
    """actually allocated specification for a job"""

    topology: Optional[Topology] = Field(description="hardware topology", default=None)
    hardware: HardwareResult = Field(description="actual allocated hardware")
    distributed: bool = Field(
        description="whether or not the run is happening over multiple hosts"
    )


class _BaseJob(ABC):
    @abstractmethod
    def __call__(self) -> None:
        """We will call this function simultaneously on all hosts"""
        None
