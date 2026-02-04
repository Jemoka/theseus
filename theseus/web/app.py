"""
Theseus Web UI - FastAPI application.

Usage:
    uvicorn theseus.web.app:app --reload --host 0.0.0.0 --port 8000

Or via CLI:
    theseus web --cluster-root /path/to/cluster/root
"""

# mypy: ignore-errors
# FastAPI lifecycle hooks have complex typing

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from theseus.web.services import (
    StatusService,
    CheckpointService,
    LogService,
    AlertService,
)
from theseus.web.routes import api, views


def create_app(
    cluster_root: Optional[Path] = None,
    debug: bool = False,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        cluster_root: Root directory of the cluster (contains status/, checkpoints/, etc.)
        debug: Enable debug mode
    """
    # Default to environment variable or current directory
    if cluster_root is None:
        cluster_root = Path(os.environ.get("THESEUS_CLUSTER_ROOT", "."))
    cluster_root = Path(cluster_root)

    # Initialize services
    status_dir = cluster_root / "status"
    checkpoints_dir = cluster_root / "checkpoints"

    status_service = StatusService(status_dir)
    checkpoint_service = CheckpointService(checkpoints_dir)
    log_service = LogService(status_dir)
    alert_service = AlertService(status_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store services in app state
        app.state.cluster_root = cluster_root
        app.state.status_service = status_service
        app.state.checkpoint_service = checkpoint_service
        app.state.log_service = log_service
        app.state.alert_service = alert_service

        # Initial alert scan
        jobs = status_service.list_all_jobs(limit=1000)
        alert_service.check_for_alerts(jobs)

        yield

    app = FastAPI(
        title="Theseus Dashboard",
        description="Job monitoring dashboard for Theseus",
        version="0.1.0",
        debug=debug,
        lifespan=lifespan,
    )

    # Setup templates
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    # Register template filters
    @app.on_event("startup")
    def setup_template_filters():
        templates.env.filters["format_size"] = checkpoint_service.format_size

    # Mount static files if directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    app.include_router(api.router, prefix="/api")
    app.include_router(views.router)

    return app


# Default app instance (uses env vars for config)
app = create_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Theseus Web UI")
    parser.add_argument(
        "--cluster-root",
        type=Path,
        default=None,
        help="Cluster root directory (default: THESEUS_CLUSTER_ROOT env var or current dir)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.cluster_root:
        os.environ["THESEUS_CLUSTER_ROOT"] = str(args.cluster_root)

    uvicorn.run(
        "theseus.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
