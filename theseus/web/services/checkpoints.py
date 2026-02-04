"""
Checkpoint service - browses checkpoint_dir for saved checkpoints.

Checkpoint directory structure (from CheckpointedJob):
    checkpoints_dir/
        {project}/
            {group}/
                {job_name}/
                    latest          # text file with latest suffix
                    {nested_dirs}/  # can be nested up to 3 levels
                        config.yaml # This marks a checkpoint directory
                        checkpoint/ # Orbax checkpoint data
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

    def _find_checkpoint_dirs(
        self,
        base_path: Path,
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> list[Path]:
        """
        Recursively find directories containing config.yaml up to max_depth.
        """
        checkpoint_dirs = []

        if current_depth >= max_depth:
            return checkpoint_dirs

        try:
            for item in base_path.iterdir():
                # Skip 'latest' file
                if item.name == "latest":
                    continue

                if item.is_dir():
                    # Check if this directory contains config.yaml
                    if (item / "config.yaml").exists():
                        checkpoint_dirs.append(item)
                    else:
                        # Recursively search subdirectories
                        checkpoint_dirs.extend(
                            self._find_checkpoint_dirs(
                                item, max_depth, current_depth + 1
                            )
                        )
        except (OSError, PermissionError):
            pass

        return checkpoint_dirs

    def _get_checkpoint_info(
        self,
        checkpoint_path: Path,
        project: str,
        group: str,
        job_name: str,
    ) -> Optional[CheckpointInfo]:
        """Parse a checkpoint directory into CheckpointInfo."""
        try:
            # Get the suffix - the relative path from job_name directory
            job_dir = self.checkpoints_dir / project / group / job_name
            suffix = str(checkpoint_path.relative_to(job_dir))

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
                has_config=(checkpoint_path / "config.yaml").exists(),
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

                    # Find all checkpoint directories (those containing config.yaml)
                    checkpoint_dirs = self._find_checkpoint_dirs(job_dir, max_depth=3)

                    for checkpoint_dir in checkpoint_dirs:
                        info = self._get_checkpoint_info(
                            checkpoint_dir,
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
        # Handle nested paths in suffix
        path = self.checkpoints_dir / project / group / job_name / suffix

        # Check if this path contains config.yaml (is a checkpoint)
        if path.exists() and (path / "config.yaml").exists():
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

        # If no latest file, get the most recent checkpoint
        checkpoints = self.list_job_checkpoints(project, group, job_name)
        if checkpoints:
            return checkpoints[0]

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
        """Read the config from a checkpoint (tries config.yaml then config.json)."""
        base_path = self.checkpoints_dir / project / group / job_name / suffix

        # Try config.yaml first
        yaml_path = base_path / "config.yaml"
        if yaml_path.exists():
            try:
                import yaml

                with open(yaml_path) as f:
                    return yaml.safe_load(f)
            except Exception:
                pass

        # Fall back to config.json
        json_path = base_path / "config.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    return json.load(f)
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
