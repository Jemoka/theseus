"""
Theseus Web UI - Job monitoring dashboard.

Usage:
    theseus web --cluster-root /path/to/cluster/root
    # or
    python -m theseus.web.app --cluster-root /path/to/cluster/root
"""


def __getattr__(name: str):
    if name == "create_app":
        from theseus.web.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app"]
