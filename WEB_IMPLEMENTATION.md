# Theseus Web UI Implementation Guide

This document describes the web dashboard implementation and provides guidance for extending it.

## Quick Start

```bash
# Install dependencies (add to pyproject.toml or requirements)
pip install fastapi uvicorn jinja2 aiofiles sse-starlette

# Run the server
THESEUS_CLUSTER_ROOT=/path/to/cluster/root uvicorn theseus.web.app:app --reload

# Or use the app directly
python -m theseus.web.app --cluster-root /path/to/cluster/root --port 8000
```

## Architecture Overview

```
theseus/web/
├── __init__.py          # Package exports
├── app.py               # FastAPI app factory
├── models.py            # Pydantic data models
├── services/            # Business logic layer
│   ├── status.py        # Job status from status_dir
│   ├── checkpoints.py   # Checkpoint browsing
│   ├── logs.py          # Log reading/streaming
│   └── alerts.py        # Alert management
├── routes/
│   ├── api.py           # REST API endpoints (/api/*)
│   └── views.py         # HTML page routes (/)
├── templates/           # Jinja2 templates
│   ├── base.html        # Base layout
│   ├── components/      # Reusable UI components
│   └── partials/        # HTMX partial renders
└── static/              # Static assets (optional)
```

## Key Features

### Implemented
- **Dashboard**: Overview of running jobs, recent activity, project stats
- **Jobs listing**: Filterable list of all job runs
- **Job detail**: Individual job view with log streaming
- **Checkpoints**: Browse saved checkpoints
- **Alerts**: Track job events (start/complete/fail/preempt)
- **HTMX polling**: Auto-refresh for running jobs and stats

### Data Sources
The UI reads from:
1. `{cluster_root}/status/` - Job metadata and logs (written by bootstrap.py)
2. `{cluster_root}/checkpoints/` - Saved model checkpoints

---

## TODO List (for you to fill in)

### High Priority

#### 1. W&B Integration
**File**: `theseus/web/services/status.py` (line ~55)

Extract wandb URL from job config and populate `job.wandb_url`:
```python
def _extract_wandb_url(self, config: dict) -> Optional[str]:
    # TODO: Extract wandb URL from config
    # Config structure: config.logging.wandb.{entity, project, ...}
    # URL format: https://wandb.ai/{entity}/{project}/runs/{run_id}
    pass
```

Then add a link in `components/job_card.html`:
```html
{% if job.wandb_url %}
<a href="{{ job.wandb_url }}" target="_blank" class="...">W&B</a>
{% endif %}
```

#### 2. Hardware Allocation View
**File**: Create `theseus/web/templates/hardware.html`

Show current hardware utilization across clusters:
- Which hosts have running jobs
- GPU/TPU allocation per host
- Available capacity

Data source: Aggregate from running jobs' `hardware` fields.

#### 3. ANSI Color Parsing for Logs
**File**: `theseus/web/templates/components/log_viewer.html`

The logs contain ANSI color codes from loguru. Options:
1. Use [ansi-to-html](https://github.com/rburns/ansi-to-html) in JS
2. Parse server-side and emit HTML spans
3. Use a library like [xterm.js](https://xtermjs.org/) for terminal emulation

#### 4. Persist Alerts
**File**: `theseus/web/services/alerts.py`

Currently alerts are in-memory only. Options:
1. Write to `{status_dir}/alerts.json`
2. Use SQLite for persistence
3. Integrate with external alerting (Slack, PagerDuty)

### Medium Priority

#### 5. Job Actions
**Files**: `routes/api.py`, `components/job_card.html`

Add ability to:
- Cancel running jobs (kill SLURM job)
- Restart failed jobs
- Delete old job metadata

#### 6. Checkpoint Actions
**Files**: `routes/api.py`, `components/checkpoint_row.html`

Add ability to:
- Delete checkpoints
- Download checkpoint config
- "Load" checkpoint (start new job from checkpoint)

#### 7. Search & Filtering
**File**: `templates/jobs.html`, `templates/checkpoints.html`

Add:
- Full-text search across job names/configs
- Date range filters
- Sort options (name, date, status)

#### 8. Pagination
**Files**: `routes/api.py`, `routes/views.py`

Currently limited to 100 items. Add proper pagination:
- Offset/limit params
- Page navigation UI
- URL state persistence

### Low Priority

#### 9. Dark Mode Toggle
**File**: `templates/base.html`

Add a toggle button to switch between light/dark themes:
```javascript
// Store preference in localStorage
// Toggle 'dark' class on <html>
```

#### 10. Real-time Log Streaming with SSE
**File**: `routes/api.py` (endpoint exists), `components/log_viewer.html`

The SSE endpoint `/api/logs/{...}/stream` is implemented but not used.
Switch from HTMX polling to SSE for more efficient streaming:
```html
<div hx-ext="sse" sse-connect="/api/logs/.../stream" sse-swap="log">
```

#### 11. Notification Badge
**File**: `templates/base.html`

Show unacknowledged alert count in nav:
```html
<a href="/alerts" class="...">
    Alerts
    {% if alert_count > 0 %}
    <span class="badge">{{ alert_count }}</span>
    {% endif %}
</a>
```

#### 12. CLI Integration
**File**: `theseus/cli.py`

Add a `web` command:
```python
@cli.command()
@click.option("--port", default=8000)
@click.option("--host", default="0.0.0.0")
def web(port: int, host: str):
    """Start the web dashboard."""
    import uvicorn
    uvicorn.run("theseus.web.app:app", host=host, port=port)
```

---

## Component Reference

### Status Badge
`templates/components/status_badge.html`

Displays job status with color coding:
- Running: blue with pulse animation
- Completed: green
- Failed: red
- Preempted: yellow

**Customization**: Edit the Tailwind classes to match your design system.

### Job Card
`templates/components/job_card.html`

Compact job summary card showing:
- Name (linked to detail)
- Status badge
- Project/group
- Hardware allocation
- Duration

**TODO sections marked** for adding:
- W&B link
- Quick action buttons

### Stat Card
`templates/components/stat_card.html`

Numeric metric display with color accent.

**TODO**: Add trend indicators, sparklines.

### Log Viewer
`templates/components/log_viewer.html`

Terminal-style log display with:
- Auto-scroll on update
- HTMX polling (3s interval for running jobs)
- Refresh button

**TODO**: ANSI colors, search, download.

### Alert Item
`templates/components/alert_item.html`

Single alert with icon, message, and acknowledge button.

---

## API Reference

All API endpoints return JSON and are prefixed with `/api`.

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/jobs` | List jobs (query: project, group, status, limit) |
| GET | `/api/jobs/running` | List running jobs |
| GET | `/api/jobs/recent?hours=24` | Jobs from last N hours |
| GET | `/api/jobs/{project}/{group}/{name}/{run_id}` | Get specific job |
| GET | `/api/jobs/{project}/{group}/{name}` | Get all runs of a job |

### Checkpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/checkpoints` | List checkpoints (query: project, group, job_name, limit) |
| GET | `/api/checkpoints/{project}/{group}/{job_name}` | List job's checkpoints |
| GET | `/api/checkpoints/{project}/{group}/{job_name}/latest` | Get latest checkpoint |
| GET | `/api/checkpoints/{project}/{group}/{job_name}/{suffix}` | Get specific checkpoint |
| GET | `/api/checkpoints/{project}/{group}/{job_name}/{suffix}/config` | Get checkpoint config |

### Logs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/logs/{project}/{group}/{name}/{run_id}?tail=100` | Get log content |
| GET | `/api/logs/{project}/{group}/{name}/{run_id}/stream` | SSE log stream |

### Alerts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | List alerts (query: alert_type, acknowledged, limit) |
| GET | `/api/alerts/recent?hours=24` | Recent alerts |
| POST | `/api/alerts/{alert_id}/acknowledge` | Acknowledge alert |
| POST | `/api/alerts/acknowledge-all` | Acknowledge all alerts |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/stats` | Aggregated statistics |
| GET | `/api/projects` | List projects with stats |
| GET | `/api/health` | Health check |

---

## Styling Guide

The UI uses Tailwind CSS via CDN. Key design decisions:

1. **Minimal base styling** - Components are functional but intentionally plain
2. **Dark mode support** - All components have `dark:` variants
3. **Semantic colors** - Status colors (blue/green/red/yellow) are consistent

### Customization Points

1. **Brand colors**: Edit `tailwind.config` in `base.html`:
   ```javascript
   tailwind.config = {
     theme: {
       extend: {
         colors: {
           brand: { 500: '#your-color' }
         }
       }
     }
   }
   ```

2. **Component styles**: Each component has CSS classes you can override

3. **Custom CSS**: Add to `static/` and include in `base.html`

### Recommended Improvements

1. Add a consistent spacing scale
2. Define typography hierarchy
3. Add subtle animations/transitions
4. Improve mobile responsiveness
5. Add loading skeletons

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
web = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "jinja2>=3.1.0",
    "aiofiles>=23.0.0",
    "sse-starlette>=1.6.0",
]
```

---

## Testing

TODO: Add tests for:
- [ ] Service layer (status, checkpoints, logs, alerts)
- [ ] API endpoints
- [ ] Template rendering

Example test structure:
```python
# tests/web/test_status_service.py
def test_list_all_jobs(tmp_path):
    # Create mock status directory structure
    # Verify service correctly reads metadata
    pass
```

---

## Deployment Notes

For production deployment:

1. **Don't use Tailwind CDN** - Build a production CSS file
2. **Add authentication** - The current UI has no auth
3. **Use proper ASGI server** - gunicorn with uvicorn workers
4. **Add CORS if needed** - For API access from other origins
5. **Consider caching** - Status/checkpoint data doesn't change often

Example production command:
```bash
gunicorn theseus.web.app:app -w 4 -k uvicorn.workers.UvicornWorker
```
