"""
In-memory job metadata cache with background refresh.

Avoids hammering the filesystem (especially on network filesystems like JuiceFS)
by caching all job metadata in memory and refreshing via background polling.

Supports two refresh modes:
  - "polling" (default): Background asyncio task periodically stats files and
    re-reads only those whose mtime changed. Safe for JuiceFS and any filesystem.
  - "watchdog": Uses the watchdog library's native OS file watcher for instant
    change detection. Falls back to polling on platforms/filesystems that lack
    inotify support. Best for local development.

Usage:
    cache = JobCache(status_dir, mode="polling", poll_interval=10.0)
    await cache.start()  # initial scan + begin background refresh
    jobs = cache.list_all_jobs(project="foo")
    ...
    await cache.stop()
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from theseus.web.models import (
    DashboardStats,
    HardwareInfo,
    JobMetadata,
    JobStatus,
    ProjectSummary,
)

logger = logging.getLogger(__name__)


class _CacheEntry:
    """A cached job plus the mtime of its metadata.json when we last read it."""

    __slots__ = ("job", "mtime", "wandb_mtime")

    def __init__(self, job: JobMetadata, mtime: float, wandb_mtime: float):
        self.job = job
        self.mtime = mtime
        self.wandb_mtime = wandb_mtime


# Type alias for cache key
_Key = tuple[str, str, str, str]  # (project, group, name, run_id)


class JobCache:
    """In-memory cache of job metadata with background refresh."""

    def __init__(
        self,
        status_dir: Path,
        mode: str = "polling",
        poll_interval: float = 10.0,
    ):
        self.status_dir = Path(status_dir)
        self.mode = mode
        self.poll_interval = poll_interval

        # Cache storage: keyed by (project, group, name, run_id)
        self._entries: dict[_Key, _CacheEntry] = {}
        self._lock = asyncio.Lock()

        # Set of known metadata paths -> key, for quick invalidation
        self._path_to_key: dict[Path, _Key] = {}

        # Background task handle
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Watchdog observer (optional, only in watchdog mode)
        self._observer = None

        # Track last full-scan time so we can do infrequent full scans
        # (catches deleted jobs that mtime-checking alone won't detect)
        self._last_full_scan: float = 0
        self._full_scan_interval: float = 60.0  # seconds between full scans

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the cache: do initial full scan, then begin background refresh."""
        logger.info(
            "JobCache starting (mode=%s, poll_interval=%.1fs)",
            self.mode,
            self.poll_interval,
        )
        await self._full_scan()
        self._running = True

        if self.mode == "watchdog":
            self._start_watchdog()

        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop background refresh."""
        self._running = False
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("JobCache stopped")

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Background loop that refreshes the cache."""
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval)
                now = time.monotonic()
                if now - self._last_full_scan >= self._full_scan_interval:
                    await self._full_scan()
                else:
                    await self._incremental_scan()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cache poll loop")

    async def _full_scan(self) -> None:
        """Walk the entire status directory and rebuild cache."""
        loop = asyncio.get_running_loop()
        entries, path_map = await loop.run_in_executor(None, self._full_scan_sync)
        async with self._lock:
            self._entries = entries
            self._path_to_key = path_map
        self._last_full_scan = time.monotonic()
        logger.debug("Full scan complete: %d jobs cached", len(entries))

    def _full_scan_sync(self) -> tuple[dict[_Key, _CacheEntry], dict[Path, _Key]]:
        """Synchronous full directory walk (runs in thread pool)."""
        entries: dict[_Key, _CacheEntry] = {}
        path_map: dict[Path, _Key] = {}

        if not self.status_dir.exists():
            return entries, path_map

        for project_dir in _safe_iterdir(self.status_dir):
            if not project_dir.is_dir():
                continue
            for group_dir in _safe_iterdir(project_dir):
                if not group_dir.is_dir():
                    continue
                for job_dir in _safe_iterdir(group_dir):
                    if not job_dir.is_dir():
                        continue
                    for run_dir in _safe_iterdir(job_dir):
                        if not run_dir.is_dir():
                            continue
                        metadata_path = run_dir / "metadata.json"
                        if not metadata_path.exists():
                            continue

                        entry = self._read_entry(metadata_path)
                        if entry is not None:
                            key = (
                                entry.job.project or "",
                                entry.job.group or "",
                                entry.job.name,
                                entry.job.run_id,
                            )
                            entries[key] = entry
                            path_map[metadata_path] = key

        return entries, path_map

    async def _incremental_scan(self) -> None:
        """Check mtimes and re-read only changed files (runs in thread pool)."""
        loop = asyncio.get_running_loop()
        updates = await loop.run_in_executor(None, self._incremental_scan_sync)
        if updates:
            async with self._lock:
                for key, entry in updates.items():
                    self._entries[key] = entry
            logger.debug("Incremental scan: %d jobs updated", len(updates))

    def _incremental_scan_sync(self) -> dict[_Key, _CacheEntry]:
        """Synchronous mtime check (runs in thread pool)."""
        updates: dict[_Key, _CacheEntry] = {}

        for metadata_path, key in list(self._path_to_key.items()):
            try:
                current_mtime = metadata_path.stat().st_mtime
            except OSError:
                continue

            old_entry = self._entries.get(key)
            if old_entry is None or current_mtime != old_entry.mtime:
                entry = self._read_entry(metadata_path)
                if entry is not None:
                    updates[key] = entry

            # Also check if wandb URL needs refreshing (log file changed)
            if old_entry is not None:
                log_path = metadata_path.parent / "output.log"
                try:
                    log_mtime = log_path.stat().st_mtime
                except OSError:
                    log_mtime = 0
                if log_mtime != old_entry.wandb_mtime and key not in updates:
                    entry = self._read_entry(metadata_path)
                    if entry is not None:
                        updates[key] = entry

        return updates

    # ------------------------------------------------------------------
    # Watchdog integration (optional)
    # ------------------------------------------------------------------

    def _start_watchdog(self) -> None:
        """Start a watchdog observer for native filesystem events."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning(
                "watchdog not installed, falling back to polling-only mode. "
                "Install with: pip install watchdog"
            )
            return

        cache = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                if event.src_path.endswith("metadata.json"):
                    cache._on_file_changed(Path(event.src_path))

            def on_created(self, event):
                if event.is_directory:
                    return
                if event.src_path.endswith("metadata.json"):
                    cache._on_file_changed(Path(event.src_path))

            def on_deleted(self, event):
                if event.is_directory:
                    return
                if event.src_path.endswith("metadata.json"):
                    cache._on_file_deleted(Path(event.src_path))

        observer = Observer()
        observer.schedule(_Handler(), str(self.status_dir), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("Watchdog observer started for %s", self.status_dir)

    def _on_file_changed(self, metadata_path: Path) -> None:
        """Handle a metadata.json change detected by watchdog."""
        entry = self._read_entry(metadata_path)
        if entry is None:
            return
        key = (
            entry.job.project or "",
            entry.job.group or "",
            entry.job.name,
            entry.job.run_id,
        )
        # Direct assignment is thread-safe for dict in CPython (GIL),
        # and the asyncio lock is only for bulk operations.
        self._entries[key] = entry
        self._path_to_key[metadata_path] = key

    def _on_file_deleted(self, metadata_path: Path) -> None:
        """Handle a metadata.json deletion detected by watchdog."""
        key = self._path_to_key.pop(metadata_path, None)
        if key is not None:
            self._entries.pop(key, None)

    # ------------------------------------------------------------------
    # Parsing helpers (synchronous, called from thread pool)
    # ------------------------------------------------------------------

    def _read_entry(self, metadata_path: Path) -> Optional[_CacheEntry]:
        """Read and parse a single metadata.json into a cache entry."""
        try:
            mtime = metadata_path.stat().st_mtime
        except OSError:
            return None

        try:
            with open(metadata_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

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
        wandb_url = _extract_wandb_url(log_path)
        try:
            wandb_mtime = log_path.stat().st_mtime
        except OSError:
            wandb_mtime = 0

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
            wandb_url=wandb_url,
        )

        return _CacheEntry(job=job, mtime=mtime, wandb_mtime=wandb_mtime)

    # ------------------------------------------------------------------
    # Public query interface (called from async route handlers)
    # ------------------------------------------------------------------

    def list_all_jobs(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> list[JobMetadata]:
        """List all jobs from cache, with optional filtering."""
        jobs: list[JobMetadata] = []
        for entry in self._entries.values():
            j = entry.job
            if project and j.project != project:
                continue
            if group and j.group != group:
                continue
            if status and j.status != status:
                continue
            jobs.append(j)

        jobs.sort(key=lambda j: j.start_time, reverse=True)
        return jobs[:limit]

    def get_job(
        self, project: str, group: str, name: str, run_id: str
    ) -> Optional[JobMetadata]:
        """Get a single job from cache."""
        entry = self._entries.get((project, group, name, run_id))
        return entry.job if entry is not None else None

    def get_job_runs(self, project: str, group: str, name: str) -> list[JobMetadata]:
        """Get all runs for a specific job name from cache."""
        jobs = [
            entry.job
            for key, entry in self._entries.items()
            if key[0] == project and key[1] == group and key[2] == name
        ]
        jobs.sort(key=lambda j: j.start_time, reverse=True)
        return jobs

    def list_projects(self) -> list[ProjectSummary]:
        """List all projects with summary stats, from cache."""
        projects: dict[str, ProjectSummary] = {}

        for entry in self._entries.values():
            j = entry.job
            project_name = j.project or "unknown"

            if project_name not in projects:
                projects[project_name] = ProjectSummary(name=project_name)

            summary = projects[project_name]
            summary.total_jobs += 1
            group_name = j.group or "default"
            if group_name not in summary.groups:
                summary.groups.append(group_name)

            if j.status == JobStatus.RUNNING:
                summary.running += 1
            elif j.status == JobStatus.COMPLETED:
                summary.completed += 1
            elif j.status == JobStatus.FAILED:
                summary.failed += 1

        for summary in projects.values():
            summary.groups.sort()

        return sorted(projects.values(), key=lambda p: p.name)

    def get_running_jobs(self) -> list[JobMetadata]:
        """Get all running, non-stale jobs from cache."""
        return [
            entry.job
            for entry in self._entries.values()
            if entry.job.status == JobStatus.RUNNING and not entry.job.is_stale
        ]

    def get_recent_jobs(self, hours: int = 24, limit: int = 50) -> list[JobMetadata]:
        """Get jobs started within the last N hours, from cache."""
        from datetime import datetime

        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent = []
        for entry in self._entries.values():
            try:
                start = datetime.fromisoformat(entry.job.start_time).timestamp()
                if start >= cutoff:
                    recent.append(entry.job)
            except Exception:
                continue

        recent.sort(key=lambda j: j.start_time)
        return recent[:limit]

    def get_dashboard_stats(self) -> DashboardStats:
        """Get aggregated stats from cache."""
        projects = self.list_projects()
        all_entries = list(self._entries.values())

        stats = DashboardStats(
            total_jobs=len(all_entries),
            projects=projects,
        )

        active_chips = 0
        for entry in all_entries:
            j = entry.job
            if j.status == JobStatus.RUNNING:
                stats.running += 1
                active_chips += j.hardware.total_chips
            elif j.status == JobStatus.COMPLETED:
                stats.completed += 1
            elif j.status == JobStatus.FAILED:
                stats.failed += 1
            elif j.status == JobStatus.PREEMPTED:
                stats.preempted += 1

        stats.active_chips = active_chips
        return stats

    def delete_job(self, project: str, group: str, name: str, run_id: str) -> None:
        """Remove a job from the cache (called after filesystem delete)."""
        key = (project, group, name, run_id)
        self._entries.pop(key, None)
        # Also clean up path_to_key
        self._path_to_key = {
            p: k for p, k in self._path_to_key.items() if k != key
        }

    def invalidate(self, project: str, group: str, name: str, run_id: str) -> None:
        """Force re-read a specific job on next incremental scan."""
        key = (project, group, name, run_id)
        entry = self._entries.get(key)
        if entry is not None:
            entry.mtime = 0  # Force mismatch on next check


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_iterdir(path: Path):
    """iterdir() that returns empty on permission/OS errors."""
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _extract_wandb_url(log_path: Path) -> Optional[str]:
    """Extract W&B URL from output.log file."""
    if not log_path.exists():
        return None
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
        content = re.sub(r"\x1b\[[0-9;]*m", "", content)
        match = re.search(r"WANDB \|.*?url=(https://wandb\.ai/[^\s]+)", content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None
