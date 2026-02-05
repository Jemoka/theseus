"""
Route modules for the web UI.

- api: JSON API endpoints
- views: HTML page routes
- auth: Authentication routes (login, logout)
"""

from theseus.web.routes import api, auth, views

__all__ = ["api", "auth", "views"]
