"""
Alert service - tracks job events and generates alerts.

Alerts are generated for:
- Job started
- Job completed
- Job failed
- Job preempted

TODO: This is a simple in-memory implementation.
Consider persisting alerts to disk or using a proper event store.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from theseus.web.models import Alert, JobMetadata, JobStatus


class AlertService:
    """Service for managing job alerts."""

    def __init__(self, status_dir: Path, max_alerts: int = 100):
        self.status_dir = Path(status_dir)
        self.max_alerts = max_alerts
        self._alerts: dict[str, Alert] = {}
        self._seen_jobs: set[str] = set()

    def _generate_alert_id(self, job: JobMetadata, event_type: str) -> str:
        """Generate a unique alert ID."""
        key = f"{job.project}:{job.group}:{job.name}:{job.run_id}:{event_type}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _job_key(self, job: JobMetadata) -> str:
        """Generate a unique key for a job run."""
        return f"{job.project}:{job.group}:{job.name}:{job.run_id}"

    def check_for_alerts(self, jobs: list[JobMetadata]) -> list[Alert]:
        """
        Check a list of jobs for new alertable events.

        Returns newly generated alerts.
        """
        new_alerts: list[Alert] = []

        for job in jobs:
            job_key = self._job_key(job)

            # Check if this is a new job we haven't seen
            if job_key not in self._seen_jobs:
                self._seen_jobs.add(job_key)

                # Generate alert based on status
                if job.status == JobStatus.RUNNING:
                    alert = self._create_alert(
                        job, "started", f"Job {job.name} started"
                    )
                elif job.status == JobStatus.FAILED:
                    alert = self._create_alert(job, "failed", f"Job {job.name} failed")
                elif job.status == JobStatus.COMPLETED:
                    alert = self._create_alert(
                        job, "completed", f"Job {job.name} completed"
                    )
                elif job.status == JobStatus.PREEMPTED:
                    alert = self._create_alert(
                        job, "preempted", f"Job {job.name} was preempted"
                    )
                else:
                    continue

                new_alerts.append(alert)

        return new_alerts

    def _create_alert(self, job: JobMetadata, event_type: str, message: str) -> Alert:
        """Create and store an alert."""
        alert_id = self._generate_alert_id(job, event_type)

        alert = Alert(
            id=alert_id,
            type=event_type,
            job_name=job.name,
            project=job.project,
            group=job.group,
            run_id=job.run_id,
            timestamp=datetime.now().isoformat(),
            message=message,
        )

        self._alerts[alert_id] = alert

        # Prune old alerts if we exceed max
        if len(self._alerts) > self.max_alerts:
            sorted_alerts = sorted(
                self._alerts.values(), key=lambda a: a.timestamp, reverse=True
            )
            self._alerts = {a.id: a for a in sorted_alerts[: self.max_alerts]}

        return alert

    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 50,
    ) -> list[Alert]:
        """
        Get alerts, optionally filtered.

        Returns alerts sorted by timestamp (most recent first).
        """
        alerts = list(self._alerts.values())

        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]

    def get_recent_alerts(self, hours: int = 24, limit: int = 20) -> list[Alert]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now().timestamp() - (hours * 3600)

        recent = []
        for alert in self._alerts.values():
            try:
                ts = datetime.fromisoformat(alert.timestamp).timestamp()
                if ts >= cutoff:
                    recent.append(alert)
            except Exception:
                continue

        recent.sort(key=lambda a: a.timestamp, reverse=True)
        return recent[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all alerts. Returns count of acknowledged."""
        count = 0
        for alert in self._alerts.values():
            if not alert.acknowledged:
                alert.acknowledged = True
                count += 1
        return count

    def clear_alerts(self) -> int:
        """Clear all alerts. Returns count cleared."""
        count = len(self._alerts)
        self._alerts.clear()
        self._seen_jobs.clear()
        return count

    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alerts."""
        return sum(1 for a in self._alerts.values() if not a.acknowledged)

    def get_failed_alerts(self, limit: int = 10) -> list[Alert]:
        """Get recent failed job alerts."""
        return self.get_alerts(alert_type="failed", limit=limit)
