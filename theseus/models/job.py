"""
Things you can train/infer/etc.
"""

from abc import ABC
from typing import Type, Any, Optional
from pydantic import BaseModel, Field, ImportString, field_validator

from theseus.models.topology import Topology
from theseus.models.hardware import HardwareRequest


class JobSpec(BaseModel):
    """user provided specification for a job"""

    name: str = Field(description="name of the job, useful for logging, etc.")
    project: Optional[str] = Field(
        description="project this run belongs to", default="theseus"
    )
    id: Optional[str] = Field(
        description="ID of the run, could be from wandb and the like, could be not yet assigned"
    )
    group: Optional[str] = Field(
        description="group under the project this run belongs to", default=""
    )
    job: ImportString[Type["_BaseJob"]]
    request: HardwareRequest

    @field_validator("job")
    @classmethod
    def must_be_job(cls, v: Any) -> Type["_BaseJob"]:
        if not isinstance(v, type):
            raise TypeError(f"Expected a class, got {type(v).__name__}")
        if not issubclass(v, _BaseJob):
            raise TypeError(f"{v} must subclass Job")
        return v


class ExecutionSpec(JobSpec):
    """actually allocated specification for a job"""

    topology: Topology = Field(description="hardware topology")
    distributed: bool = Field(
        description="whether or not the run is happening over multiple hosts"
    )


class _BaseJob(ABC): ...
