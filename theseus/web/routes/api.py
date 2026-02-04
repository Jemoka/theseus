"""
REST API endpoints for the web UI.

All endpoints return JSON and are prefixed with /api.
"""

# mypy: ignore-errors
# FastAPI route handlers have complex typing that mypy doesn't handle well

from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from theseus.web.models import (
    JobMetadata,
    JobStatus,
    CheckpointInfo,
    ProjectSummary,
    DashboardStats,
)

router = APIRouter(tags=["api"])


# === Jobs ===


@router.get("/jobs", response_model=list[JobMetadata])
async def list_jobs(
    request: Request,
    project: Optional[str] = None,
    group: Optional[str] = None,
    status: Optional[JobStatus] = None,
    limit: int = Query(default=100, le=1000),
):
    """List all jobs with optional filtering."""
    service = request.app.state.status_service
    return service.list_all_jobs(
        project=project, group=group, status=status, limit=limit
    )


@router.get("/jobs/running", response_model=list[JobMetadata])
async def list_running_jobs(request: Request):
    """List all currently running jobs."""
    service = request.app.state.status_service
    return service.get_running_jobs()


@router.get("/jobs/recent", response_model=list[JobMetadata])
async def list_recent_jobs(
    request: Request,
    hours: int = Query(default=24, le=168),
    limit: int = Query(default=50, le=500),
):
    """List jobs started within the last N hours."""
    service = request.app.state.status_service
    return service.get_recent_jobs(hours=hours, limit=limit)


@router.get("/jobs/{project}/{group}/{name}/{run_id}", response_model=JobMetadata)
async def get_job(
    request: Request,
    project: str,
    group: str,
    name: str,
    run_id: str,
):
    """Get a specific job by its identifiers."""
    service = request.app.state.status_service
    job = service.get_job(project, group, name, run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{project}/{group}/{name}", response_model=list[JobMetadata])
async def get_job_runs(
    request: Request,
    project: str,
    group: str,
    name: str,
):
    """Get all runs for a specific job."""
    service = request.app.state.status_service
    return service.get_job_runs(project, group, name)


# === Projects ===


@router.get("/projects", response_model=list[ProjectSummary])
async def list_projects(request: Request):
    """List all projects with summary stats."""
    service = request.app.state.status_service
    return service.list_projects()


# === Dashboard ===


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(request: Request):
    """Get aggregated dashboard statistics."""
    service = request.app.state.status_service
    checkpoint_service = request.app.state.checkpoint_service

    stats = service.get_dashboard_stats()
    stats.total_checkpoints = checkpoint_service.count_checkpoints()

    return stats


# === Checkpoints ===


@router.get(
    "/checkpoints/{project}/{group}/{job_name}", response_model=list[CheckpointInfo]
)
async def list_job_checkpoints(
    request: Request,
    project: str,
    group: str,
    job_name: str,
):
    """List all checkpoints for a specific job."""
    service = request.app.state.checkpoint_service
    return service.list_job_checkpoints(project, group, job_name)


# === Logs ===


@router.get("/logs/{project}/{group}/{name}/{run_id}")
async def get_log(
    request: Request,
    project: str,
    group: str,
    name: str,
    run_id: str,
    tail: Optional[int] = Query(default=None, description="Return last N lines"),
    offset: int = Query(default=0, description="Byte offset to start reading from"),
):
    """
    Get log contents for a job.

    If tail is specified, returns the last N lines.
    Otherwise, returns content starting from offset.
    """
    service = request.app.state.log_service

    if not service.log_exists(project, group, name, run_id):
        raise HTTPException(status_code=404, detail="Log not found")

    if tail is not None:
        content = service.tail_log(project, group, name, run_id, lines=tail)
        return {"content": content, "offset": None}

    content, new_offset = service.read_log(project, group, name, run_id, offset=offset)
    return {"content": content, "offset": new_offset}


@router.get("/logs/{project}/{group}/{name}/{run_id}/stream")
async def stream_log(
    request: Request,
    project: str,
    group: str,
    name: str,
    run_id: str,
):
    """
    Stream log contents via Server-Sent Events.

    Connect to this endpoint for real-time log updates.
    """
    service = request.app.state.log_service

    if not service.log_exists(project, group, name, run_id):
        raise HTTPException(status_code=404, detail="Log not found")

    async def event_generator():
        async for content in service.stream_log(project, group, name, run_id):
            yield {"event": "log", "data": content}

    return EventSourceResponse(event_generator())


# === Health ===


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    cluster_root = request.app.state.cluster_root
    return {
        "status": "healthy",
        "cluster_root": str(cluster_root),
        "status_dir_exists": (cluster_root / "status").exists(),
        "checkpoints_dir_exists": (cluster_root / "checkpoints").exists(),
    }
