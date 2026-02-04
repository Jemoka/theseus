"""
Log service - handles log file reading and streaming.

Supports:
- Reading log files
- Tailing log files (last N lines)
- Streaming logs via Server-Sent Events (SSE)
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiofiles  # type: ignore[import-untyped]


class LogService:
    """Service for reading and streaming job logs."""

    def __init__(self, status_dir: Path):
        self.status_dir = Path(status_dir)

    def _get_log_path(self, project: str, group: str, name: str, run_id: str) -> Path:
        """Get the log file path for a job."""
        return self.status_dir / project / group / name / run_id / "output.log"

    def read_log(
        self,
        project: str,
        group: str,
        name: str,
        run_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> tuple[str, int]:
        """
        Read log file contents.

        Args:
            project: Project name
            group: Group name
            name: Job name
            run_id: Run ID
            offset: Byte offset to start reading from
            limit: Maximum bytes to read (None for all)

        Returns:
            Tuple of (content, new_offset)
        """
        log_path = self._get_log_path(project, group, name, run_id)

        if not log_path.exists():
            return "", 0

        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(offset)
                if limit:
                    content = f.read(limit)
                else:
                    content = f.read()
                new_offset = f.tell()

            return content, new_offset
        except Exception as e:
            return f"Error reading log: {e}", offset

    def tail_log(
        self,
        project: str,
        group: str,
        name: str,
        run_id: str,
        lines: int = 100,
    ) -> str:
        """
        Read the last N lines of a log file.

        This is a simple implementation that reads the whole file.
        TODO: Optimize for large files by reading from end.
        """
        log_path = self._get_log_path(project, group, name, run_id)

        if not log_path.exists():
            return ""

        try:
            with open(log_path, "r", errors="replace") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading log: {e}"

    async def stream_log(
        self,
        project: str,
        group: str,
        name: str,
        run_id: str,
        poll_interval: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """
        Stream log file contents as they're written.

        Yields new content whenever the file is updated.
        Uses polling - TODO: consider inotify/fsevents for efficiency.
        """
        log_path = self._get_log_path(project, group, name, run_id)
        offset = 0

        while True:
            if not log_path.exists():
                await asyncio.sleep(poll_interval)
                continue

            try:
                async with aiofiles.open(log_path, "r") as f:
                    await f.seek(offset)
                    content = await f.read()
                    if content:
                        offset = await f.tell()
                        yield content
            except Exception:
                pass

            await asyncio.sleep(poll_interval)

    def get_log_size(self, project: str, group: str, name: str, run_id: str) -> int:
        """Get the size of a log file in bytes."""
        log_path = self._get_log_path(project, group, name, run_id)
        if log_path.exists():
            return log_path.stat().st_size
        return 0

    def log_exists(self, project: str, group: str, name: str, run_id: str) -> bool:
        """Check if a log file exists."""
        return self._get_log_path(project, group, name, run_id).exists()
