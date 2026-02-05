"""
HTML view routes for the web UI.

These routes render Jinja2 templates.
Uses HTMX for dynamic updates without full page reloads.
"""

# mypy: ignore-errors
# FastAPI route handlers have complex typing that mypy doesn't handle well

from typing import Any, Optional

from fastapi import APIRouter, Request, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from theseus.web.models import JobStatus

router = APIRouter(tags=["views"])


def render(request: Request, template: str, **context):
    """Helper to render a template with common context."""
    templates = request.app.state.templates
    context["request"] = request

    # Add common context
    context.setdefault("cluster_root", str(request.app.state.cluster_root))

    return templates.TemplateResponse(template, context)


# === Main Pages ===


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    status_service = request.app.state.status_service

    stats = status_service.get_dashboard_stats()

    running_jobs = status_service.get_running_jobs()
    # Sort running jobs by most recent heartbeat first
    running_jobs.sort(
        key=lambda j: j.last_heartbeat if j.last_heartbeat else "", reverse=True
    )

    # Get recent jobs (sorted by most recent heartbeat)
    all_jobs = status_service.list_all_jobs(limit=100)
    all_jobs.sort(
        key=lambda j: j.last_heartbeat if j.last_heartbeat else "", reverse=True
    )
    recent_jobs = all_jobs[:8]

    return render(
        request,
        "index.html",
        stats=stats,
        running_jobs=running_jobs,
        recent_jobs=recent_jobs,
    )


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_list(
    request: Request,
    project: Optional[str] = None,
    group: Optional[str] = None,
    status: Optional[str] = None,
):
    """Jobs listing page."""
    status_service = request.app.state.status_service

    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            pass

    jobs = status_service.list_all_jobs(project=project, group=group, status=job_status)
    projects = status_service.list_projects()

    return render(
        request,
        "jobs.html",
        jobs=jobs,
        projects=projects,
        current_project=project,
        current_group=group,
        current_status=status,
    )


@router.get("/jobs/{project}/{group}/{name}/{run_id}", response_class=HTMLResponse)
async def job_detail(
    request: Request,
    project: str,
    group: str,
    name: str,
    run_id: str,
):
    """Job detail page with log viewer."""
    status_service = request.app.state.status_service
    log_service = request.app.state.log_service
    checkpoint_service = request.app.state.checkpoint_service

    job = status_service.get_job(project, group, name, run_id)
    if not job:
        return render(request, "404.html", message="Job not found")

    # Get all runs of this job, sorted by start time (most recent first)
    all_runs = status_service.get_job_runs(project, group, name)
    all_runs.sort(key=lambda r: r.start_time if r.start_time else "", reverse=True)

    # Get checkpoints for this job
    checkpoints = checkpoint_service.list_job_checkpoints(project, group, name)

    # Get initial log tail
    log_content = ""
    if log_service.log_exists(project, group, name, run_id):
        log_content = log_service.tail_log(project, group, name, run_id, lines=100)

    return render(
        request,
        "job_detail.html",
        job=job,
        all_runs=all_runs,
        current_run_id=run_id,
        checkpoints=checkpoints,
        log_content=log_content,
    )


@router.post("/jobs/{project}/{group}/{name}/{run_id}/delete")
async def delete_job(
    request: Request,
    project: str,
    group: str,
    name: str,
    run_id: str,
    confirmation: str = Form(...),
):
    """Delete a job run."""
    status_service = request.app.state.status_service

    # Check confirmation matches job name
    if confirmation != name:
        return HTMLResponse(
            content="<div style='color: red; padding: 10px;'>Confirmation failed. Job name did not match.</div>",
            status_code=400,
        )

    success = status_service.delete_job(project, group, name, run_id)

    if success:
        # Redirect to dashboard
        return RedirectResponse(url="/", status_code=303)
    else:
        return HTMLResponse(
            content="<div style='color: red; padding: 10px;'>Failed to delete job.</div>",
            status_code=500,
        )


@router.get(
    "/checkpoints/{project}/{group}/{job_name}/{suffix:path}",
    response_class=HTMLResponse,
)
async def checkpoint_detail(
    request: Request,
    project: str,
    group: str,
    job_name: str,
    suffix: str,
):
    """Checkpoint detail page."""
    checkpoint_service = request.app.state.checkpoint_service

    checkpoint = checkpoint_service.get_checkpoint(project, group, job_name, suffix)
    if not checkpoint:
        return render(request, "404.html", message="Checkpoint not found")

    config = checkpoint_service.get_checkpoint_config(project, group, job_name, suffix)
    job_spec = checkpoint_service.get_checkpoint_job_spec(
        project, group, job_name, suffix
    )

    # Get other checkpoints for this job
    other_checkpoints = checkpoint_service.list_job_checkpoints(
        project, group, job_name
    )
    other_checkpoints = [c for c in other_checkpoints if c.suffix != suffix][:10]

    return render(
        request,
        "checkpoint_detail.html",
        checkpoint=checkpoint,
        config=config,
        job_spec=job_spec,
        other_checkpoints=other_checkpoints,
    )


@router.get("/projects/{project}", response_class=HTMLResponse)
async def project_detail(request: Request, project: str):
    """Project detail page."""
    status_service = request.app.state.status_service

    jobs = status_service.list_all_jobs(project=project)

    # Group jobs by group name
    groups: dict[str, list[Any]] = {}
    for job in jobs:
        g = job.group or "default"
        if g not in groups:
            groups[g] = []
        groups[g].append(job)

    return render(
        request,
        "project_detail.html",
        project=project,
        jobs=jobs,
        groups=groups,
    )


# === HTMX Partials ===
# These return partial HTML for HTMX updates


@router.get("/partials/running-jobs", response_class=HTMLResponse)
async def partial_running_jobs(request: Request):
    """Partial: running jobs list (for HTMX polling)."""
    status_service = request.app.state.status_service
    jobs = status_service.get_running_jobs()
    # Sort by most recent heartbeat first
    jobs.sort(key=lambda j: j.last_heartbeat if j.last_heartbeat else "", reverse=True)

    return render(request, "partials/running_jobs.html", jobs=jobs)


@router.get("/partials/stats", response_class=HTMLResponse)
async def partial_stats(request: Request):
    """Partial: dashboard stats (for HTMX polling)."""
    status_service = request.app.state.status_service

    stats = status_service.get_dashboard_stats()

    return render(request, "partials/stats.html", stats=stats)


@router.get(
    "/partials/log/{project}/{group}/{name}/{run_id}", response_class=HTMLResponse
)
async def partial_log(
    request: Request,
    project: str,
    group: str,
    name: str,
    run_id: str,
    tail: int = Query(default=50),
):
    """Partial: log content (for HTMX polling)."""
    log_service = request.app.state.log_service
    content = log_service.tail_log(project, group, name, run_id, lines=tail)

    return render(request, "partials/log_content.html", content=content)
