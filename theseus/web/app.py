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
from datetime import datetime

from fastapi import Depends, FastAPI, Request as FastAPIRequest
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from theseus.web.services import (
    StatusService,
    CheckpointService,
    LogService,
)
from theseus.web.routes import api, views, auth as auth_routes
from theseus.web.auth import require_auth


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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store services in app state
        app.state.cluster_root = cluster_root
        app.state.status_service = status_service
        app.state.checkpoint_service = checkpoint_service
        app.state.log_service = log_service

        yield

    app = FastAPI(
        title="Theseus Dashboard",
        description="Job monitoring dashboard for Theseus",
        version="0.1.0",
        debug=debug,
        lifespan=lifespan,
    )

    # Add session middleware
    secret_key = os.environ.get("THESEUS_SECRET_KEY")
    if not secret_key:
        import secrets as secrets_module
        import warnings

        secret_key = secrets_module.token_urlsafe(32)
        warnings.warn(
            f"No THESEUS_SECRET_KEY set, using random key: {secret_key}\n"
            "Sessions will be invalidated on restart. Set THESEUS_SECRET_KEY for production."
        )

    app.add_middleware(SessionMiddleware, secret_key=secret_key)

    # Add auth redirect middleware
    @app.middleware("http")
    async def auth_redirect_middleware(request: FastAPIRequest, call_next):
        """Catch 401 errors and redirect to login page."""
        response = await call_next(request)

        # If 401 and not already on login/logout page, redirect to login
        if response.status_code == 401 and request.url.path not in [
            "/login",
            "/logout",
        ]:
            return RedirectResponse(url="/login", status_code=303)

        return response

    # Setup templates
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    # Register template filters
    templates.env.filters["format_size"] = checkpoint_service.format_size

    def format_timestamp(timestamp_str: Optional[str]) -> str:
        """Format ISO timestamp to a more readable format."""
        if not timestamp_str:
            return "-"

        try:
            # Handle different timestamp formats
            if "T" in timestamp_str:
                # ISO format like 2026-02-04T09:44:35
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                # Space-separated format like 2026-02-04 09:44:35
                dt = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")

            # Format as: February 4, 2026, 9:44 AM
            # Use platform-safe format (%-d doesn't work on Windows)
            formatted = dt.strftime("%B %d, %Y, %I:%M %p")
            # Remove leading zeros from day and time
            formatted = formatted.replace(
                " 0", " ", 1
            )  # Only replace first occurrence for day
            return formatted
        except (ValueError, TypeError, AttributeError):
            # Fallback: just return the first 19 chars
            return timestamp_str[:19] if len(timestamp_str) >= 19 else timestamp_str

    templates.env.filters["format_timestamp"] = format_timestamp

    def format_duration(duration_str: Optional[str]) -> str:
        """Format duration to a more readable format."""
        if not duration_str:
            return "-"

        # Clean up the duration string
        duration_str = duration_str.strip()

        # Already in a nice format like "4h 34m"? Return as-is
        if "h" in duration_str or "m" in duration_str or "s" in duration_str:
            return duration_str

        # Try to parse as seconds or other format
        try:
            # If it's a number, treat as seconds
            if duration_str.replace(".", "").isdigit():
                seconds = float(duration_str)
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)

                if hours > 0:
                    return f"{hours}h {minutes}m"
                elif minutes > 0:
                    return f"{minutes}m {secs}s"
                else:
                    return f"{secs}s"
        except (ValueError, TypeError):
            pass

        return duration_str

    templates.env.filters["format_duration"] = format_duration

    # Mount static files if directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers (auth routes first, no protection)
    app.include_router(auth_routes.router)
    # Protected routes
    app.include_router(api.router, prefix="/api", dependencies=[Depends(require_auth)])
    app.include_router(views.router, dependencies=[Depends(require_auth)])

    return app


# Default app instance (uses env vars for config)
app = create_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Theseus Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    uvicorn.run(
        "theseus.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
