"""
Pydantic models for the web API.

These models define the data structures used by the API and templates.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"
    UNKNOWN = "unknown"


class HardwareInfo(BaseModel):
    """Hardware allocation info for a job."""

    chip: Optional[str] = None
    total_chips: int = 0
    hosts: list[str] = Field(default_factory=list)


class JobMetadata(BaseModel):
    """Job metadata as written by bootstrap.py."""

    name: str
    project: Optional[str] = None
    group: Optional[str] = None
    job_key: str
    run_id: str
    start_time: str
    last_heartbeat: str
    status: JobStatus
    slurm_job_id: Optional[str] = None
    hardware: HardwareInfo
    config: dict[str, Any] = Field(default_factory=dict)

    # Computed fields (added by service layer)
    log_path: Optional[str] = None
    metadata_path: Optional[str] = None
    wandb_url: Optional[str] = None

    @property
    def duration(self) -> Optional[str]:
        """Human-readable duration since start."""
        try:
            start = datetime.fromisoformat(self.start_time)
            now = datetime.now()
            delta = now - start
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            return f"{seconds}s"
        except Exception:
            return None

    @property
    def is_stale(self) -> bool:
        """Check if heartbeat is stale (>5 minutes)."""
        try:
            heartbeat = datetime.fromisoformat(self.last_heartbeat)
            delta = datetime.now() - heartbeat
            return delta.total_seconds() > 300
        except Exception:
            return True


class CheckpointInfo(BaseModel):
    """Checkpoint metadata."""

    path: str
    suffix: str  # e.g., "step_1000", "best"
    job_name: str
    project: str
    group: str
    created: Optional[str] = None
    size_bytes: Optional[int] = None
    has_config: bool = False
    has_job_spec: bool = False


class ProjectSummary(BaseModel):
    """Summary of a project's jobs."""

    name: str
    total_jobs: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    groups: list[str] = Field(default_factory=list)


class ClusterInfo(BaseModel):
    """Cluster configuration info."""

    name: str
    root: str
    work: str
    status_dir: str
    checkpoints_dir: str
    data_dir: str
    results_dir: str


class Alert(BaseModel):
    """Alert for job events."""

    id: str
    type: str  # "failed", "started", "completed", "preempted"
    job_name: str
    project: Optional[str] = None
    group: Optional[str] = None
    run_id: str
    timestamp: str
    message: str
    acknowledged: bool = False


class DashboardStats(BaseModel):
    """Dashboard summary statistics."""

    total_jobs: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    preempted: int = 0
    total_checkpoints: int = 0
    active_chips: int = 0
    projects: list[ProjectSummary] = Field(default_factory=list)
    recent_alerts: list[Alert] = Field(default_factory=list)
