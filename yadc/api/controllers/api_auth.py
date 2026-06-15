"""``/api/auth/password`` — set/clear the ``yadc_password`` session cookie.

The cookie carries the password-mode decryption password on every
password-requiring request (``/api/envs/.../reveal``, ``/api/envs/<n>/models``,
``/api/datasets/.../caption``, etc.). Browsers auto-attach it, so the
client doesn't have to thread the password through request bodies.

The ``POST /auth/password`` endpoint validates the password *before*
setting the cookie. Validation is done by attempting to decrypt the
RSA private key via :class:`PasswordKeyStorage` — the private key is
the one thing the password always protects, regardless of which envs
exist or whether they have encrypted tokens. This is more reliable
than validating against an env, which would silently succeed for any
wrong password when the env has no encrypted tokens to decrypt.

The server-side ``YADC_PASSWORD`` env var is unaffected — it's the
fallback for requests that don't carry a cookie (e.g. server-side
automation). See :func:`_password.resolve_request_password`.
"""

import pydantic
from quart import Response, request

from yadc.cmd import config as cmd_config
from yadc.cmd.envs.keystorage_password import (
    PasswordKeyStorage,
    PasswordRequiredError,
)

from ..modules.logging_factory import LoggingFactory
from . import controller
from ._password import clear_password_cookie, set_password_cookie
from .blueprints import ApiBlueprint
from .utils_json import ErrorCode, jsonify_error, validate_body


class SetPasswordBody(pydantic.BaseModel):
    """Body for ``POST /api/auth/password`` — the new session password."""

    password: str


@controller
def api_auth(app: ApiBlueprint, logging: LoggingFactory):
    _logger = logging.get_logger(__name__)

    @app.post("/auth/password")
    async def set_password() -> Response:  # pyright: ignore[reportUnusedFunction]
        """Set the ``yadc_password`` session cookie.

        Validates the password by attempting to decrypt the
        password-mode private key. The cookie is only set on success,
        so the client knows the password works before retrying the
        original request.

        Returns:
            204 No Content + ``Set-Cookie: yadc_password=...`` on success.

        Errors:
            400 BAD_REQUEST — key storage is in keyring mode, or no
                password-mode private key has been generated yet (the
                user needs to switch modes via ``PUT /envs/key-mode`` first).
            403 PASSWORD_REQUIRED — the password is incorrect.
        """
        body = validate_body(SetPasswordBody, await request.get_json(silent=True))
        config = cmd_config.load_config()

        if config.key_storage.mode != "password":
            return jsonify_error(
                "Password auth is not active — key storage is in keyring mode.",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        storage = PasswordKeyStorage(config, password=body.password)
        if not storage.has_private_key():
            return jsonify_error(
                "Password auth is not active — no password-mode private key has been generated yet. Set a password via Settings → Security first.",
                status=400,
                code=ErrorCode.BAD_REQUEST,
            )

        try:
            storage.load_private_key()
        except PasswordRequiredError:
            _logger.warning("Wrong password supplied to /api/auth/password.")
            return jsonify_error(
                "Password is incorrect.",
                status=403,
                code=ErrorCode.PASSWORD_REQUIRED,
            )

        resp = Response(status=204)
        set_password_cookie(resp, body.password, secure=request.is_secure)
        _logger.info("Password cookie set.")
        return resp

    @app.delete("/auth/password")
    async def clear_password() -> Response:  # pyright: ignore[reportUnusedFunction]
        """Clear the ``yadc_password`` session cookie.

        Idempotent — returns 204 whether or not the cookie was set.
        """
        resp = Response(status=204)
        clear_password_cookie(resp, secure=request.is_secure)
        _logger.info("Password cookie cleared.")
        return resp
