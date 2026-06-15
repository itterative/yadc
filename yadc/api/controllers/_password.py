"""Shared helpers for the ``yadc_password`` session cookie.

The cookie carries the password-mode decryption password on every
password-requiring API request. The server-side ``YADC_PASSWORD``
env var is the fallback when no cookie is present (e.g. server-side
automation that doesn't go through the browser).

This module is private to the ``yadc.api.controllers`` package —
other modules shouldn't import from it.
"""

from quart import Request, Response

from yadc.core.env import YADC_PASSWORD

PASSWORD_COOKIE_NAME = "yadc_password"
"""Name of the session cookie that carries the decryption password."""

PASSWORD_COOKIE_PATH = "/api"
"""Cookie path — narrower than ``/`` so the cookie isn't sent to the
static-frontend handler."""


def resolve_request_password(request: Request) -> str | None:
    """Return the decryption password for this request.

    Reads the ``yadc_password`` cookie first, then falls back to the
    server-side ``YADC_PASSWORD`` env var. Mirrors the fallback chain
    in :func:`yadc.cmd.envs.encryption._resolve_password`.
    """
    cookie_pw = request.cookies.get(PASSWORD_COOKIE_NAME)
    if cookie_pw is not None:
        return cookie_pw
    return YADC_PASSWORD


def set_password_cookie(response: Response, password: str, *, secure: bool) -> None:
    """Attach the ``yadc_password`` session cookie to *response*.

    *secure* should reflect the request's actual scheme (``request.is_secure``)
    so the cookie is marked ``Secure`` in production (HTTPS) and still
    works in local dev (HTTP).
    """
    response.set_cookie(
        PASSWORD_COOKIE_NAME,
        password,
        httponly=True,
        samesite="Strict",
        secure=secure,
        path=PASSWORD_COOKIE_PATH,
    )


def clear_password_cookie(response: Response, *, secure: bool) -> None:
    """Remove the ``yadc_password`` cookie from the browser.

    The *path* must match the one used at set time or the browser
    won't actually delete the cookie.
    """
    response.delete_cookie(
        PASSWORD_COOKIE_NAME,
        path=PASSWORD_COOKIE_PATH,
        secure=secure,
    )
