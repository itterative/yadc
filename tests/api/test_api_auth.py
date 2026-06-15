"""Tests for the ``/api/auth/password`` endpoints — set/clear the
``yadc_password`` session cookie.

The auth endpoint validates the password *before* setting the cookie
by attempting to decrypt the password-mode private key via
:class:`PasswordKeyStorage`. These tests mock the storage layer so the
controller's branching logic is exercised without actually generating
an RSA keypair.
"""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from quart import Quart

from yadc.api.configuration import Configuration
from yadc.api.controllers.api_auth import api_auth
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.cmd.envs.keystorage_password import PasswordRequiredError

# Patch targets for the controller module. Single source of truth — change
# these here if the controller's imports move.
_PATCH_CMD_CONFIG = "yadc.api.controllers.api_auth.cmd_config"
_PATCH_PASSWORD_KEY_STORAGE = "yadc.api.controllers.api_auth.PasswordKeyStorage"


def _fake_config(*, mode: Literal["keyring", "password"] = "password"):
    """Build a minimal config mock with the requested key storage mode."""
    config = MagicMock()
    config.key_storage.mode = mode
    return config


@pytest.fixture
def client(test_configuration: Configuration):
    """Quart test client with the auth blueprint registered."""
    app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_auth(bp, mock_logging)

    app.register_blueprint(bp)
    return app.test_client()


@pytest.fixture
def patched_cmd_config():
    """Patch the ``cmd_config`` module in the controller."""
    with patch(_PATCH_CMD_CONFIG) as mock_config:
        yield mock_config


@pytest.fixture
def patched_password_key_storage():
    """Patch the ``PasswordKeyStorage`` class in the controller.

    The patch returns a mock that, when called as a constructor,
    returns a mock storage instance with ``has_private_key()`` and
    ``load_private_key()`` attributes that the test can configure.
    """
    with patch(_PATCH_PASSWORD_KEY_STORAGE) as mock_class:
        # Make the constructor return a fresh storage mock each call.
        mock_class.return_value = MagicMock()
        yield mock_class


class TestSetPassword:
    """``POST /api/auth/password`` — validates the password and sets the
    ``yadc_password`` session cookie on success."""

    @pytest.mark.asyncio
    async def test_returns_204_on_valid_password(self, client, patched_cmd_config, patched_password_key_storage):
        """Right password → 204 + ``Set-Cookie`` header with the password.

        ``PasswordKeyStorage.load_private_key()`` is the canonical
        "is the password correct?" check. When it returns bytes
        (decryption succeeded), the cookie is set.
        """
        patched_cmd_config.load_config.return_value = _fake_config(mode="password")
        storage = patched_password_key_storage.return_value
        storage.has_private_key.return_value = True
        storage.load_private_key.return_value = b"decrypted-pem-bytes"

        resp = await client.post("/api/auth/password", json={"password": "hunter2"})

        assert resp.status_code == 204
        # Verify the cookie was set with the right name and value.
        set_cookie = resp.headers.get("Set-Cookie", "")
        assert "yadc_password=hunter2" in set_cookie
        # Cookie hardening — HttpOnly + SameSite=Strict are required.
        assert "HttpOnly" in set_cookie
        assert "SameSite=Strict" in set_cookie

    @pytest.mark.asyncio
    async def test_returns_403_on_wrong_password(self, client, patched_cmd_config, patched_password_key_storage):
        """Wrong password → ``PasswordKeyStorage.load_private_key()`` raises
        ``PasswordRequiredError`` → 403 ``PASSWORD_REQUIRED``.

        No ``Set-Cookie`` header is attached (the cookie is only set
        on success).
        """
        patched_cmd_config.load_config.return_value = _fake_config(mode="password")
        storage = patched_password_key_storage.return_value
        storage.has_private_key.return_value = True
        storage.load_private_key.side_effect = PasswordRequiredError("test")

        resp = await client.post("/api/auth/password", json={"password": "wrong"})

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["code"] == "PASSWORD_REQUIRED"
        assert "incorrect" in data["error"].lower()
        # No cookie set on failure.
        assert "yadc_password" not in resp.headers.get("Set-Cookie", "")

    @pytest.mark.asyncio
    async def test_returns_400_when_keyring_mode(self, client, patched_cmd_config, patched_password_key_storage):
        """``key_storage.mode == "keyring"`` → 400 BAD_REQUEST.

        Password auth is meaningless in keyring mode (the password is
        never consulted). Refuse to set a cookie that would do
        nothing.
        """
        patched_cmd_config.load_config.return_value = _fake_config(mode="keyring")

        resp = await client.post("/api/auth/password", json={"password": "anything"})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        # Storage should NOT be constructed or consulted.
        patched_password_key_storage.assert_not_called()
        assert "yadc_password" not in resp.headers.get("Set-Cookie", "")

    @pytest.mark.asyncio
    async def test_returns_400_when_no_private_key(self, client, patched_cmd_config, patched_password_key_storage):
        """Password mode but no private key on disk → 400 BAD_REQUEST.

        The user hasn't set up password mode yet (no key has been
        generated). They need to switch modes via
        ``PUT /envs/key-mode`` first, which generates the key pair
        with their password.
        """
        patched_cmd_config.load_config.return_value = _fake_config(mode="password")
        storage = patched_password_key_storage.return_value
        storage.has_private_key.return_value = False

        resp = await client.post("/api/auth/password", json={"password": "anything"})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        # load_private_key must NOT be called when has_private_key is False.
        storage.load_private_key.assert_not_called()
        assert "yadc_password" not in resp.headers.get("Set-Cookie", "")

    @pytest.mark.asyncio
    async def test_missing_body_returns_400(self, client, patched_cmd_config):
        """Empty/missing body fails Pydantic validation → 400 BAD_REQUEST."""
        resp = await client.post("/api/auth/password", json={})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"


class TestClearPassword:
    """``DELETE /api/auth/password`` — clears the session cookie."""

    @pytest.mark.asyncio
    async def test_returns_204(self, client):
        """Idempotent — returns 204 whether or not the cookie was set."""
        resp = await client.delete("/api/auth/password")

        assert resp.status_code == 204
        # The deletion response sets a cookie with Max-Age=0 / empty value
        # to instruct the browser to remove it. Path must match the set
        # path (/api) or the browser won't actually delete it.
        set_cookie = resp.headers.get("Set-Cookie", "")
        assert "yadc_password" in set_cookie
        assert "Path=/api" in set_cookie
