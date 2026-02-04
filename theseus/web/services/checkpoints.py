"""
Checkpoint service - browses checkpoint_dir for saved checkpoints.

Checkpoint directory structure (from CheckpointedJob):
    checkpoints_dir/
        {project}/
            {group}/
                {job_name}/
                    latest          # text file with latest suffix
                    {suffix}/       # e.g., step_1000, best
                        checkpoint/ # Orbax checkpoint data
                        config.json
                        config.yaml
                        job.json
                        rng.npy
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from theseus.web.models import CheckpointInfo


class CheckpointService:
    """Service for browsing checkpoints."""

    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = Path(checkpoints_dir)

    def _get_checkpoint_info(
        self,
        checkpoint_path: Path,
        project: str,
        group: str,
        job_name: str,
    ) -> Optional[CheckpointInfo]:
        """Parse a checkpoint directory into CheckpointInfo."""
        try:
            suffix = checkpoint_path.name

            # Get creation time from directory mtime
            stat = checkpoint_path.stat()
            created = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Calculate total size
            total_size = 0
            for root, dirs, files in os.walk(checkpoint_path):
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

            return CheckpointInfo(
                path=str(checkpoint_path),
                suffix=suffix,
                job_name=job_name,
                project=project,
                group=group,
                created=created,
                size_bytes=total_size,
                has_config=(checkpoint_path / "config.json").exists(),
                has_job_spec=(checkpoint_path / "job.json").exists(),
            )
        except Exception:
            return None

    def list_all_checkpoints(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        job_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[CheckpointInfo]:
        """
        List all checkpoints, optionally filtered.

        Returns checkpoints sorted by creation time (most recent first).
        """
        checkpoints: list[CheckpointInfo] = []

        if not self.checkpoints_dir.exists():
            return checkpoints

        for project_dir in self.checkpoints_dir.iterdir():
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
                    if job_name and job_dir.name != job_name:
                        continue

                    for item in job_dir.iterdir():
                        # Skip 'latest' file
                        if item.name == "latest":
                            continue
                        if not item.is_dir():
                            continue

                        info = self._get_checkpoint_info(
                            item,
                            project=project_dir.name,
                            group=group_dir.name,
                            job_name=job_dir.name,
                        )
                        if info:
                            checkpoints.append(info)

        # Sort by creation time (most recent first)
        checkpoints.sort(key=lambda c: c.created or "", reverse=True)

        return checkpoints[:limit]

    def get_checkpoint(
        self, project: str, group: str, job_name: str, suffix: str
    ) -> Optional[CheckpointInfo]:
        """Get a specific checkpoint."""
        path = self.checkpoints_dir / project / group / job_name / suffix
        if path.exists():
            return self._get_checkpoint_info(path, project, group, job_name)
        return None

    def get_latest_checkpoint(
        self, project: str, group: str, job_name: str
    ) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint for a job."""
        job_dir = self.checkpoints_dir / project / group / job_name
        latest_file = job_dir / "latest"

        if latest_file.exists():
            try:
                suffix = latest_file.read_text().strip()
                return self.get_checkpoint(project, group, job_name, suffix)
            except Exception:
                pass

        return None

    def list_job_checkpoints(
        self, project: str, group: str, job_name: str
    ) -> list[CheckpointInfo]:
        """List all checkpoints for a specific job."""
        return self.list_all_checkpoints(
            project=project, group=group, job_name=job_name, limit=1000
        )

    def get_checkpoint_config(
        self, project: str, group: str, job_name: str, suffix: str
    ) -> Optional[dict[str, Any]]:
        """Read the config.json from a checkpoint."""
        path = (
            self.checkpoints_dir / project / group / job_name / suffix / "config.json"
        )
        if path.exists():
            try:
                with open(path) as f:
                    result: dict[str, Any] = json.load(f)
                    return result
            except Exception:
                pass
        return None

    def get_checkpoint_job_spec(
        self, project: str, group: str, job_name: str, suffix: str
    ) -> Optional[dict[str, Any]]:
        """Read the job.json from a checkpoint."""
        path = self.checkpoints_dir / project / group / job_name / suffix / "job.json"
        if path.exists():
            try:
                with open(path) as f:
                    result: dict[str, Any] = json.load(f)
                    return result
            except Exception:
                pass
        return None

    def count_checkpoints(self) -> int:
        """Count total number of checkpoints."""
        return len(self.list_all_checkpoints(limit=100000))

    def get_total_size(self) -> int:
        """Get total size of all checkpoints in bytes."""
        total = 0
        for ckpt in self.list_all_checkpoints(limit=100000):
            if ckpt.size_bytes:
                total += ckpt.size_bytes
        return total

    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
