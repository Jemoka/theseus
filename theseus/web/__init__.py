"""
Theseus Web UI - Job monitoring dashboard.

Usage:
    theseus web --cluster-root /path/to/cluster/root
    # or
    python -m theseus.web.app --cluster-root /path/to/cluster/root
"""

from theseus.web.app import create_app

__all__ = ["create_app"]
