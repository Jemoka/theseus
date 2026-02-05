"""
Authentication for the web UI.

Uses session-based auth with login page and bcrypt password hashing.
Credentials are set via environment variables.
"""

import os
import secrets
from typing import Annotated

import bcrypt
from fastapi import Depends, Request


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception:
        return False


def get_credentials_from_env() -> tuple[str | None, str | None]:
    """
    Get username and hashed password from environment.

    Set these environment variables:
    - THESEUS_WEB_USERNAME (e.g., "admin")
    - THESEUS_WEB_PASSWORD_HASH (bcrypt hash of the password)

    Or for development, you can set:
    - THESEUS_WEB_PASSWORD (plain text - will be hashed automatically)
    """
    username = os.environ.get("THESEUS_WEB_USERNAME")
    password_hash = os.environ.get("THESEUS_WEB_PASSWORD_HASH")

    # Allow plain password for development (NOT for production!)
    plain_password = os.environ.get("THESEUS_WEB_PASSWORD")
    if plain_password and not password_hash:
        password_hash = hash_password(plain_password)

    return username, password_hash


def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate a user with username and password.

    Returns True if valid, False otherwise.
    """
    expected_username, expected_password_hash = get_credentials_from_env()

    # If no credentials are configured, allow access (warn in logs)
    if not expected_username or not expected_password_hash:
        import warnings

        warnings.warn(
            "No authentication credentials configured! "
            "Set THESEUS_WEB_USERNAME and THESEUS_WEB_PASSWORD_HASH environment variables."
        )
        return True

    # Verify username (constant-time comparison)
    username_correct = secrets.compare_digest(
        username.encode("utf-8"), expected_username.encode("utf-8")
    )

    # Verify password
    password_correct = verify_password(password, expected_password_hash)

    return username_correct and password_correct


def get_current_user(request: Request) -> str | None:
    """Get the current logged-in user from session."""
    return request.session.get("username")


def require_auth(request: Request) -> str:
    """
    Dependency that requires authentication.

    Raises HTTPException if not authenticated (handled by middleware).
    Returns username if authenticated.
    """
    username = get_current_user(request)
    if not username:
        # Check if auth is required
        expected_username, expected_password_hash = get_credentials_from_env()
        if expected_username and expected_password_hash:
            # Not authenticated - this will be caught by middleware
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )
        else:
            # No auth configured, allow access
            return "anonymous"
    return username


# Dependency for routes that require authentication
RequireAuth = Annotated[str, Depends(require_auth)]
