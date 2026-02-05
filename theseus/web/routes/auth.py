"""
Authentication routes - login, logout.
"""

# mypy: ignore-errors
# FastAPI route handlers have complex typing that mypy doesn't handle well

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from theseus.web.auth import authenticate_user, get_credentials_from_env

router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    """Show login page."""
    # Check if auth is required
    expected_username, expected_password_hash = get_credentials_from_env()
    if not expected_username or not expected_password_hash:
        # No auth configured, redirect to home
        return RedirectResponse(url="/", status_code=303)

    # Check if already logged in
    if request.session.get("username"):
        return RedirectResponse(url="/", status_code=303)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": error}
    )


@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission."""
    if authenticate_user(username, password):
        request.session["username"] = username
        return RedirectResponse(url="/", status_code=303)
    else:
        # Redirect back to login with error
        return RedirectResponse(url="/login?error=invalid", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    """Logout and clear session."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)
