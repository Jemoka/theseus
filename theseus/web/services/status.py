"""
Status service - reads job metadata from status_dir.

The status directory structure (created by bootstrap.py):
    status_dir/
        {project}/
            {group}/
                {job_name}/
                    {run_id}/
                        metadata.json
                        output.log
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from theseus.web.models import (
    JobMetadata,
    JobStatus,
    HardwareInfo,
    ProjectSummary,
    DashboardStats,
)


class StatusService:
    """Service for reading job status from the filesystem."""

    def __init__(self, status_dir: Path):
        self.status_dir = Path(status_dir)

    def _extract_wandb_url(self, log_path: Path) -> Optional[str]:
        """Extract W&B URL from output.log file."""
        if not log_path.exists():
            return None

        try:
            # Read log and search for W&B line
            # Format: WANDB | project=... run_id=... url=https://wandb.ai/...
            with open(log_path, "r", errors="ignore") as f:
                content = f.read()

            # Remove ANSI codes first
            content = re.sub(r"\x1b\[[0-9;]*m", "", content)

            # Extract W&B URL from structured log line
            match = re.search(r"WANDB \|.*?url=(https://wandb\.ai/[^\s]+)", content)
            if match:
                return match.group(1)

        except Exception:
            pass

        return None

    def _parse_metadata(self, metadata_path: Path) -> Optional[JobMetadata]:
        """Parse a metadata.json file into JobMetadata."""
        try:
            with open(metadata_path) as f:
                data = json.load(f)

            hardware_data = data.get("hardware", {})
            hardware = HardwareInfo(
                chip=hardware_data.get("chip"),
                total_chips=hardware_data.get("total_chips", 0),
                hosts=hardware_data.get("hosts", []),
            )

            status_str = data.get("status", "unknown")
            try:
                status = JobStatus(status_str)
            except ValueError:
                status = JobStatus.UNKNOWN

            log_path = metadata_path.parent / "output.log"
            job = JobMetadata(
                name=data.get("name", "unknown"),
                project=data.get("project"),
                group=data.get("group"),
                job_key=data.get("job_key", "unknown"),
                run_id=data.get("run_id", "unknown"),
                start_time=data.get("start_time", ""),
                last_heartbeat=data.get("last_heartbeat", ""),
                status=status,
                slurm_job_id=data.get("slurm_job_id"),
                hardware=hardware,
                config=data.get("config", {}),
                metadata_path=str(metadata_path),
                log_path=str(log_path),
            )

            # Extract W&B URL from output.log if available
            job.wandb_url = self._extract_wandb_url(log_path)

            return job
        except Exception:
            # TODO: Log parsing errors
            return None

    def list_all_jobs(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> list[JobMetadata]:
        """
        List all jobs, optionally filtered by project/group/status.

        Jobs are returned sorted by start_time (most recent first).
        """
        jobs: list[JobMetadata] = []

        if not self.status_dir.exists():
            return jobs

        # Traverse: status_dir / project / group / job_name / run_id / metadata.json
        for project_dir in self.status_dir.iterdir():
            if not project_dir.is_dir():
                continue
            if project and project_dir.name != project:
                continue

            for group_dir in project_dir.iterdir():
                if not group_dir.is_dir():
                    continue
                if group and group_dir.name != group:
                    continue

                for job_dir in group_dir.iterdir():
                    if not job_dir.is_dir():
                        continue

                    for run_dir in job_dir.iterdir():
                        if not run_dir.is_dir():
                            continue

                        metadata_path = run_dir / "metadata.json"
                        if metadata_path.exists():
                            job = self._parse_metadata(metadata_path)
                            if job:
                                if status and job.status != status:
                                    continue
                                jobs.append(job)

        # Sort by start time (most recent first)
        jobs.sort(key=lambda j: j.start_time, reverse=True)

        return jobs[:limit]

    def get_job(
        self, project: str, group: str, name: str, run_id: str
    ) -> Optional[JobMetadata]:
        """Get a specific job by its identifiers."""
        metadata_path = (
            self.status_dir / project / group / name / run_id / "metadata.json"
        )
        if metadata_path.exists():
            return self._parse_metadata(metadata_path)
        return None

    def get_job_runs(self, project: str, group: str, name: str) -> list[JobMetadata]:
        """Get all runs for a specific job name."""
        jobs: list[JobMetadata] = []
        job_dir = self.status_dir / project / group / name

        if not job_dir.exists():
            return jobs

        for run_dir in job_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                job = self._parse_metadata(metadata_path)
                if job:
                    jobs.append(job)

        jobs.sort(key=lambda j: j.start_time, reverse=True)
        return jobs

    def list_projects(self) -> list[ProjectSummary]:
        """List all projects with summary stats."""
        projects: dict[str, ProjectSummary] = {}

        if not self.status_dir.exists():
            return []

        for project_dir in self.status_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_name = project_dir.name
            summary = ProjectSummary(name=project_name)
            groups_seen: set[str] = set()

            for group_dir in project_dir.iterdir():
                if not group_dir.is_dir():
                    continue
                groups_seen.add(group_dir.name)

                for job_dir in group_dir.iterdir():
                    if not job_dir.is_dir():
                        continue

                    for run_dir in job_dir.iterdir():
                        metadata_path = run_dir / "metadata.json"
                        if metadata_path.exists():
                            job = self._parse_metadata(metadata_path)
                            if job:
                                summary.total_jobs += 1
                                if job.status == JobStatus.RUNNING:
                                    summary.running += 1
                                elif job.status == JobStatus.COMPLETED:
                                    summary.completed += 1
                                elif job.status == JobStatus.FAILED:
                                    summary.failed += 1

            summary.groups = sorted(groups_seen)
            projects[project_name] = summary

        return sorted(projects.values(), key=lambda p: p.name)

    def get_running_jobs(self) -> list[JobMetadata]:
        """Get all currently running jobs, excluding stale ones (no heartbeat >5min)."""
        jobs = self.list_all_jobs(status=JobStatus.RUNNING, limit=1000)
        return [j for j in jobs if not j.is_stale]

    def get_recent_jobs(self, hours: int = 24, limit: int = 50) -> list[JobMetadata]:
        """Get jobs started within the last N hours."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        jobs = self.list_all_jobs(limit=1000)

        recent = []
        for job in jobs:
            try:
                start = datetime.fromisoformat(job.start_time).timestamp()
                if start >= cutoff:
                    recent.append(job)
            except Exception:
                continue

        # Sort by start time (earliest first)
        recent.sort(key=lambda j: j.start_time)
        return recent[:limit]

    def get_dashboard_stats(self) -> DashboardStats:
        """Get aggregated stats for the dashboard."""
        jobs = self.list_all_jobs(limit=10000)
        projects = self.list_projects()

        stats = DashboardStats(
            total_jobs=len(jobs),
            projects=projects,
        )

        active_chips = 0
        for job in jobs:
            if job.status == JobStatus.RUNNING:
                stats.running += 1
                active_chips += job.hardware.total_chips
            elif job.status == JobStatus.COMPLETED:
                stats.completed += 1
            elif job.status == JobStatus.FAILED:
                stats.failed += 1
            elif job.status == JobStatus.PREEMPTED:
                stats.preempted += 1

        stats.active_chips = active_chips

        return stats

    def delete_job(self, project: str, group: str, name: str, run_id: str) -> bool:
        """Delete a specific job run. Returns True if successful."""
        run_dir = self.status_dir / project / group / name / run_id
        if run_dir.exists() and run_dir.is_dir():
            try:
                shutil.rmtree(run_dir)

                # Clean up empty parent directories
                job_dir = run_dir.parent
                if job_dir.exists() and not any(job_dir.iterdir()):
                    job_dir.rmdir()
                    group_dir = job_dir.parent
                    if group_dir.exists() and not any(group_dir.iterdir()):
                        group_dir.rmdir()
                        project_dir = group_dir.parent
                        if project_dir.exists() and not any(project_dir.iterdir()):
                            project_dir.rmdir()

                return True
            except Exception:
                return False
        return False
