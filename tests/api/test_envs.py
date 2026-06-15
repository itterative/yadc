"""Tests for the /api/envs/* endpoints."""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quart import Quart

from yadc.api.configuration import Configuration
from yadc.api.controllers.api_envs import api_envs
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.cmd.config import AppConfigEnv, AppConfigEnvValue
from yadc.cmd.envs.keystorage_password import PasswordRequiredError

# Patch targets for the controller module. Single source of truth — change
# these here if the controller's imports move.
_PATCH_CMD_CONFIG = "yadc.api.controllers.api_envs.cmd_config"
_PATCH_CMD_ENVS = "yadc.api.controllers.api_envs.cmd_envs"
_PATCH_SET_KEY_MODE = "yadc.api.controllers.api_envs.cmd_envs.set_key_mode"
_PATCH_YADC_PASSWORD = "yadc.api.controllers.api_envs.YADC_PASSWORD"
_PATCH_LIST_MODELS = "yadc.api.controllers.api_envs.cmd_envs.list_models"
# The `_password` helper module owns the YADC_PASSWORD fallback. The
# controller's own `YADC_PASSWORD` import (above) is only used by the
# `GET /envs/key-mode` endpoint to surface `env_password_set` in the
# response; the helper is what `resolve_request_password` actually
# reads for the fallback chain.
_PATCH_RESOLVE_PASSWORD_FALLBACK = "yadc.api.controllers._password.YADC_PASSWORD"


def make_app_config_env(
    *,
    api_url: str = "http://localhost:11434",
    api_token: str | None = "token",
    api_model_name: str | None = "gemma3",
    max_concurrent: int | None = None,
    encryption_method: Literal["none", "keyring", "password"] = "none",
) -> AppConfigEnv:
    """Build an ``AppConfigEnv`` with sensible defaults.

    Defaults model the common "happy path" env: a localhost Ollama URL,
    a plaintext token, a model name, and no encryption. Override any
    field to test a different shape (e.g. ``api_model_name=None`` for
    the "no default" case, or ``encryption_method="password"`` for an
    encrypted token).
    """
    return AppConfigEnv(
        api_url=AppConfigEnvValue(value=api_url, method="none"),
        api_token=AppConfigEnvValue(value=api_token, method=encryption_method),
        api_model_name=AppConfigEnvValue(value=api_model_name, method="none"),
        max_concurrent=max_concurrent,
    )


@pytest.fixture
def client(test_configuration: Configuration):
    """Quart test client with the envs blueprint registered."""
    app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    api_envs(bp, test_configuration, mock_logging)

    app.register_blueprint(bp)
    return app.test_client()


@pytest.fixture
def patched_cmd_config():
    """Patch the ``cmd_config`` module in the controller."""
    with patch(_PATCH_CMD_CONFIG) as mock_config:
        yield mock_config


@pytest.fixture
def patched_cmd_envs():
    """Patch the ``cmd_envs`` module in the controller."""
    with patch(_PATCH_CMD_ENVS) as mock_cmd_envs:
        yield mock_cmd_envs


@pytest.fixture
def patched_set_key_mode():
    """Patch ``cmd_envs.set_key_mode`` in the controller."""
    with patch(_PATCH_SET_KEY_MODE) as mock:
        yield mock


@pytest.fixture
def yadc_password():
    """Return a context manager that patches ``YADC_PASSWORD`` in the
    controller to a given value. Used by the GET /key-mode tests, where
    the env password's presence is part of the response."""

    def _patch(value):
        return patch(_PATCH_YADC_PASSWORD, value)

    return _patch


@pytest.fixture
def cookie_password():
    """Factory fixture: set the ``yadc_password`` cookie on the test client.

    The cookie path is ``/api`` (matching the production cookie config),
    so it's only sent to API routes — not the static frontend. Server
    name is ``localhost`` (the test client default).
    """

    def _set(client, value):
        client.set_cookie("localhost", "yadc_password", value, path="/api")

    return _set


@pytest.fixture
def patched_resolve_password_fallback():
    """Factory fixture: patch ``YADC_PASSWORD`` in the ``_password`` helper
    so the ``resolve_request_password`` fallback chain returns *value*
    when no cookie is set. The patched attribute is replaced with the
    string directly (``new=``) — not a MagicMock — so the resolved
    password is exactly the string the controller forwards to
    ``cmd_envs.list_models``."""

    def _patch(value):
        return patch(_PATCH_RESOLVE_PASSWORD_FALLBACK, new=value)

    return _patch


@pytest.fixture
def patched_list_models():
    """Patch ``cmd_envs.list_models`` in the controller."""
    with patch(_PATCH_LIST_MODELS, new_callable=AsyncMock) as mock:
        yield mock


class TestGetKeyMode:
    """GET /api/envs/key-mode — returns the current storage mode and whether
    YADC_PASSWORD is set in the backend environment."""

    @pytest.mark.asyncio
    async def test_returns_password_mode_when_env_password_not_set(self, client, patched_cmd_config, yadc_password):
        patched_cmd_config.load_config.return_value.key_storage.mode = "password"
        with yadc_password(None):
            resp = await client.get("/api/envs/key-mode")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["mode"] == "password"
        assert data["env_password_set"] is False

    @pytest.mark.asyncio
    async def test_returns_keyring_mode_when_env_password_set(self, client, patched_cmd_config, yadc_password):
        patched_cmd_config.load_config.return_value.key_storage.mode = "keyring"
        with yadc_password("secret"):
            resp = await client.get("/api/envs/key-mode")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["mode"] == "keyring"
        assert data["env_password_set"] is True


class TestPutKeyMode:
    """PUT /api/envs/key-mode — switches storage mode or changes the password
    for password mode. Wraps ``cmd_envs.set_key_mode`` and maps domain errors
    to the right HTTP status codes."""

    @pytest.mark.asyncio
    async def test_wrong_old_password_returns_403_password_required(self, client, patched_cmd_config, patched_set_key_mode, cookie_password):
        """Wrong current password returns 403 PASSWORD_REQUIRED, not 500 INTERNAL_ERROR.

        The backend distinguishes "you typed the wrong current password" (user
        error) from "something exploded on the server" (server error). The
        frontend relies on the 403 + PASSWORD_REQUIRED code to show a clear
        "Current password is incorrect" message instead of a generic 500.

        The current password comes from the ``yadc_password`` session cookie.
        """
        patched_cmd_config.load_config.return_value.key_storage.mode = "password"
        patched_set_key_mode.side_effect = PasswordRequiredError("Password-protected private key requires a password to decrypt.")
        cookie_password(client, "wrong")

        resp = await client.put(
            "/api/envs/key-mode",
            json={"mode": "password", "password": "newpass"},
        )

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["code"] == "PASSWORD_REQUIRED"
        assert "incorrect" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_password_change_returns_200(self, client, patched_cmd_config, patched_set_key_mode, cookie_password):
        """Current password comes from the cookie; new password from the body."""
        patched_cmd_config.load_config.return_value.key_storage.mode = "password"
        cookie_password(client, "oldpass")

        resp = await client.put(
            "/api/envs/key-mode",
            json={"mode": "password", "password": "newpass"},
        )

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["mode"] == "password"
        patched_set_key_mode.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_400_bad_request(self, client, patched_cmd_config):
        """Invalid mode value is rejected before any cmd_envs work happens."""
        resp = await client.put(
            "/api/envs/key-mode",
            json={"mode": "invalid", "password": "x"},
        )

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"


class TestPutEnv:
    """PUT /api/envs/<name> — create or update an environment.

    The endpoint distinguishes between three states per field:

    - **omitted from body**: leave the existing value untouched
    - **string value**: store the new value (token is re-encrypted)
    - **explicit null**: clear the field (mirrors ``yadc envs delete <key>``)

    Pydantic's ``model_fields_set`` is the source of truth for "what the
    client actually sent" — we must not use ``model_dump(exclude_none=True)``
    because that conflates "absent" with "null" and breaks clearing.
    """

    @pytest.mark.asyncio
    async def test_sets_all_three_fields(self, client, patched_cmd_config, patched_cmd_envs):
        """Happy path — all three fields in the body are forwarded to
        ``cmd_envs.update_env`` with the right env name."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put(
            "/api/envs/default",
            json={"api_url": "https://api.example.com", "api_token": "sk-abc", "api_model_name": "gpt-4o"},
        )

        assert resp.status_code == 200
        # update_env is called once per field that was in the body
        calls = patched_cmd_envs.update_env.call_args_list
        assert len(calls) == 3
        by_key = {c.args[0]: c.args[1] for c in calls}
        assert by_key == {
            "api_url": "https://api.example.com",
            "api_token": "sk-abc",
            "api_model_name": "gpt-4o",
        }
        for c in calls:
            assert c.kwargs == {"env": "default", "config": patched_cmd_config.load_config.return_value}
        patched_cmd_envs.save_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_body_only_updates_provided_fields(self, client, patched_cmd_config, patched_cmd_envs):
        """A field omitted from the body must not be touched. This is the
        contract that lets the frontend update one field at a time without
        clobbering the others (e.g. editing the URL shouldn't blank the
        token)."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"api_url": "https://api.example.com"})

        assert resp.status_code == 200
        calls = patched_cmd_envs.update_env.call_args_list
        assert len(calls) == 1
        assert calls[0].args == ("api_url", "https://api.example.com")
        assert calls[0].kwargs["env"] == "default"

    @pytest.mark.asyncio
    async def test_empty_body_is_a_no_op_save(self, client, patched_cmd_config, patched_cmd_envs):
        """An empty body still triggers a ``save_env`` (a no-op write) but
        must not call ``update_env`` for any field. This is the contract
        that prevents accidentally wiping fields when the client sends
        `{}` by mistake."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={})

        assert resp.status_code == 200
        patched_cmd_envs.update_env.assert_not_called()
        patched_cmd_envs.save_env.assert_called_once()

    @pytest.mark.asyncio
    async def test_null_api_model_name_clears_the_model(self, client, patched_cmd_config, patched_cmd_envs):
        """Regression: ``api_model_name: null`` must reach
        ``cmd_envs.update_env`` as ``None`` so the existing default model
        is cleared. Previously the endpoint used
        ``body.model_dump(exclude_none=True)`` which silently dropped
        ``None`` values, making it impossible to clear a model name from
        the WebUI."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"api_model_name": None})

        assert resp.status_code == 200
        patched_cmd_envs.update_env.assert_called_once()
        _args, kwargs = patched_cmd_envs.update_env.call_args
        assert _args == ("api_model_name", None)
        assert kwargs["env"] == "default"

    @pytest.mark.asyncio
    async def test_null_api_url_clears_the_url(self, client, patched_cmd_config, patched_cmd_envs):
        """Same null-clears semantics for ``api_url``."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"api_url": None})

        assert resp.status_code == 200
        patched_cmd_envs.update_env.assert_called_once_with("api_url", None, env="default", config=patched_cmd_config.load_config.return_value)

    @pytest.mark.asyncio
    async def test_null_api_token_clears_the_token(self, client, patched_cmd_config, patched_cmd_envs):
        """Same null-clears semantics for ``api_token`` (the encrypted
        field). Confirms clearing works uniformly across plain and
        encrypted fields."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"api_token": None})

        assert resp.status_code == 200
        patched_cmd_envs.update_env.assert_called_once_with("api_token", None, env="default", config=patched_cmd_config.load_config.return_value)

    @pytest.mark.asyncio
    async def test_mixed_set_and_null_in_one_request(self, client, patched_cmd_config, patched_cmd_envs):
        """Setting one field and clearing another in the same request
        must work — the two cases go through the same loop body."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put(
            "/api/envs/default",
            json={"api_url": "https://new.example.com", "api_model_name": None},
        )

        assert resp.status_code == 200
        calls = {c.args[0]: c.args[1] for c in patched_cmd_envs.update_env.call_args_list}
        assert calls == {"api_url": "https://new.example.com", "api_model_name": None}

    @pytest.mark.asyncio
    async def test_extra_field_returns_400(self, client, patched_cmd_config, patched_cmd_envs):
        """Unknown fields are rejected (the body model uses
        ``extra="forbid"``) so typos don't silently no-op."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"api_url": "x", "unknown_field": "y"})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        patched_cmd_envs.update_env.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_type_returns_400(self, client, patched_cmd_config, patched_cmd_envs):
        """Non-string values (e.g. a number for ``api_url``) are rejected
        by Pydantic before any ``update_env`` call."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"api_url": 123})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        patched_cmd_envs.update_env.assert_not_called()

    @pytest.mark.asyncio
    async def test_response_reflects_saved_state(self, client, patched_cmd_config, patched_cmd_envs):
        """After a successful PUT, the response body is the formatted env
        (same shape as ``GET /envs/<name>``) so the frontend can refresh
        its local state without a second round-trip."""
        saved_env = make_app_config_env(api_url="https://new.example.com", api_model_name="gpt-4o")
        patched_cmd_envs.get_env.return_value = saved_env

        resp = await client.put(
            "/api/envs/default",
            json={"api_url": "https://new.example.com", "api_model_name": "gpt-4o"},
        )

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["name"] == "default"
        assert data["api_url"] == "https://new.example.com"
        assert data["api_model_name"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_max_concurrent_set(self, client, patched_cmd_config, patched_cmd_envs):
        """``max_concurrent: 4`` is forwarded to ``cmd_envs.update_env``."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"max_concurrent": 4})

        assert resp.status_code == 200
        patched_cmd_envs.update_env.assert_called_once_with("max_concurrent", 4, env="default", config=patched_cmd_config.load_config.return_value)

    @pytest.mark.asyncio
    async def test_max_concurrent_null_clears(self, client, patched_cmd_config, patched_cmd_envs):
        """``max_concurrent: null`` clears the env's concurrency default."""
        patched_cmd_envs.get_env.return_value = make_app_config_env()
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"max_concurrent": None})

        assert resp.status_code == 200
        patched_cmd_envs.update_env.assert_called_once_with("max_concurrent", None, env="default", config=patched_cmd_config.load_config.return_value)

    @pytest.mark.asyncio
    async def test_max_concurrent_zero_rejected(self, client, patched_cmd_config, patched_cmd_envs):
        """``max_concurrent: 0`` is rejected by Pydantic (``Field(ge=1)``).

        0 is not a valid concurrency — it would deadlock the runner's
        semaphore. The CLI's click option and the API both reject it
        at the boundary rather than letting it reach the runner.
        """
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"max_concurrent": 0})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        patched_cmd_envs.update_env.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_concurrent_negative_rejected(self, client, patched_cmd_config, patched_cmd_envs):
        """Negative values are rejected by Pydantic (``Field(ge=1)``)."""
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"max_concurrent": -1})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        patched_cmd_envs.update_env.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_concurrent_non_integer_rejected(self, client, patched_cmd_config, patched_cmd_envs):
        """Non-integer values (e.g. a string) are rejected by Pydantic."""
        patched_cmd_envs.update_env = MagicMock()

        resp = await client.put("/api/envs/default", json={"max_concurrent": "four"})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        patched_cmd_envs.update_env.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_response_includes_max_concurrent(self, client, patched_cmd_envs):
        """The GET /envs response exposes ``max_concurrent`` so the
        WebUI can prefill the edit dialog."""
        patched_cmd_envs.list_all_env.return_value = ["default"]
        patched_cmd_envs.get_env.return_value = make_app_config_env(max_concurrent=4)

        resp = await client.get("/api/envs")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data[0]["max_concurrent"] == 4

    @pytest.mark.asyncio
    async def test_get_response_max_concurrent_null_when_unset(self, client, patched_cmd_envs):
        """``max_concurrent`` is ``null`` in the response when unset
        (matches the ``int | None`` contract — distinguish unset from
        a configured value of 1)."""
        patched_cmd_envs.list_all_env.return_value = ["default"]
        patched_cmd_envs.get_env.return_value = make_app_config_env()  # max_concurrent=None

        resp = await client.get("/api/envs")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data[0]["max_concurrent"] is None


class TestListModels:
    """GET /api/envs/<name>/models — returns the env's available model list.

    GET-only since the decryption password is read from the
    ``yadc_password`` session cookie (with the ``YADC_PASSWORD`` env-var
    fallback). The browser auto-attaches the cookie so no body is
    needed.

    Delegates the actual HTTP call to ``cmd_envs.list_models`` (which uses
    the captioner system). The controller is a thin wrapper that maps
    domain errors to the right HTTP status codes and adds the env's
    configured ``api_model_name`` as the default in the response.
    """

    @pytest.mark.asyncio
    async def test_returns_404_when_env_not_found(self, client, patched_cmd_envs):
        patched_cmd_envs.get_env.return_value = None

        resp = await client.get("/api/envs/missing/models")

        assert resp.status_code == 404
        data = await resp.get_json()
        assert data["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_returns_400_when_env_has_no_api_url(self, client, patched_cmd_envs, patched_list_models):
        """A misconfigured env (no api_url) is a client-fix problem, not an
        upstream error — distinguish it from connection failures by
        returning 400 BAD_REQUEST. The orchestrator must NOT be called.
        """
        env_data = make_app_config_env(api_url="")

        patched_cmd_envs.get_env.return_value = env_data

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        assert "API URL" in data["error"]
        patched_list_models.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_403_when_token_is_password_encrypted(self, client, patched_cmd_envs, patched_list_models):
        """Without a cookie or ``YADC_PASSWORD`` env-var, a password-mode
        env's token can't be decrypted and the call surfaces a
        403 ``PASSWORD_REQUIRED``."""
        env_data = make_app_config_env(api_token="encrypted-token", encryption_method="password")

        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = PasswordRequiredError("test")

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["error"] == "Password required to decrypt environment settings"
        assert data["code"] == "PASSWORD_REQUIRED"

    @pytest.mark.asyncio
    async def test_returns_502_on_connection_error(self, client, patched_cmd_envs, patched_list_models):
        import httpx

        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = httpx.ConnectError("connection refused")

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 502
        data = await resp.get_json()
        assert data["code"] == "UPSTREAM_ERROR"

    @pytest.mark.asyncio
    async def test_returns_502_on_upstream_http_error(self, client, patched_cmd_envs, patched_list_models):
        import httpx

        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data

        request = httpx.Request("GET", "http://localhost:11434/models")
        response = httpx.Response(401, request=request)
        patched_list_models.side_effect = httpx.HTTPStatusError("unauthorized", request=request, response=response)

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 502
        data = await resp.get_json()
        assert data["code"] == "UPSTREAM_ERROR"
        assert "401" in data["error"]

    @pytest.mark.asyncio
    async def test_returns_502_on_value_error(self, client, patched_cmd_envs, patched_list_models):
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = ValueError("bad response shape")

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 502
        data = await resp.get_json()
        assert data["code"] == "UPSTREAM_ERROR"
        assert data["error"] == "bad response shape"

    @pytest.mark.asyncio
    async def test_returns_504_on_timeout(self, client, patched_cmd_envs, patched_list_models, test_configuration: Configuration):
        """A dead/slow env that exceeds ``Configuration.list_models_timeout``
        surfaces as 504 GATEWAY_TIMEOUT (not 502 UPSTREAM_ERROR) so the
        frontend can render a "timed out" message instead of a generic
        upstream error.
        """
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.side_effect = TimeoutError("simulated hang")

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 504
        data = await resp.get_json()
        assert data["code"] == "GATEWAY_TIMEOUT"
        assert str(test_configuration.list_models_timeout) in data["error"]
        assert "API URL" in data["error"] or "network" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_passes_timeout_from_configuration(self, client, patched_cmd_envs, patched_list_models, test_configuration: Configuration):
        """The ``timeout`` kwarg forwarded to ``cmd_envs.list_models``
        matches ``Configuration.list_models_timeout`` so a misconfigured
        env can't override the cap by accident.
        """
        env_data = make_app_config_env(api_model_name=None)
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = []

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("timeout") == test_configuration.list_models_timeout

    @pytest.mark.asyncio
    async def test_returns_models_and_default(self, client, patched_cmd_envs, patched_list_models):
        """Happy path — env has api_model_name set, captioner returns sorted models."""
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["a-model", "b-model", "c-model"]

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["models"] == ["a-model", "b-model", "c-model"]
        assert data["default"] == "gemma3"

    @pytest.mark.asyncio
    async def test_omits_default_when_no_model_name_configured(self, client, patched_cmd_envs, patched_list_models):
        env_data = make_app_config_env(api_model_name=None)
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1", "m2"]

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["models"] == ["m1", "m2"]
        assert "default" not in data

    @pytest.mark.asyncio
    async def test_passes_cache_ttl_from_configuration(self, client, patched_cmd_envs, patched_list_models, test_configuration: Configuration):
        """The cache_ttl kwarg forwarded to ``cmd_envs.list_models`` matches
        ``Configuration.api_models_cache_ttl``."""
        env_data = make_app_config_env(api_model_name=None)
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = []

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        # first positional arg is the env name; cache_ttl is a kwarg
        assert kwargs.get("cache_ttl") == test_configuration.api_models_cache_ttl
        # And the configured value should be the default 5-minute TTL.
        assert test_configuration.api_models_cache_ttl == 300.0

    @pytest.mark.asyncio
    async def test_default_timeout_matches_captioner_constant(self, test_configuration: Configuration):
        """Sanity: the configuration default is the shared captioner constant.

        Guards against accidental drift between the configuration
        default and the captioner package's default.
        """
        from yadc.captioners.api.constants import DEFAULT_LIST_MODELS_TIMEOUT_SECONDS

        assert test_configuration.list_models_timeout == DEFAULT_LIST_MODELS_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_cookie_password_forwarded_to_cmd_envs(self, client, patched_cmd_envs, patched_list_models, cookie_password):
        """The cookie value is forwarded to ``cmd_envs.list_models`` as the
        ``password`` kwarg so the env's token can be decrypted when
        ``YADC_PASSWORD`` is unset."""
        env_data = make_app_config_env(api_token="encrypted-token", encryption_method="password")
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1"]
        cookie_password(client, "hunter2")

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("password") == "hunter2"

    @pytest.mark.asyncio
    async def test_get_without_cookie_uses_none(self, client, patched_cmd_envs, patched_list_models):
        """No cookie means ``password=None`` is forwarded to
        ``cmd_envs.list_models``, which lets it fall back to
        ``YADC_PASSWORD`` (or fail with ``PasswordRequiredError`` if
        neither is set)."""
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1"]

        resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("password") is None

    @pytest.mark.asyncio
    async def test_falls_back_to_yadc_password_env(self, client, patched_cmd_envs, patched_list_models, patched_resolve_password_fallback):
        """No cookie + ``YADC_PASSWORD`` env var set → the env-var value
        is forwarded to ``cmd_envs.list_models``. Confirms the fallback
        chain in ``resolve_request_password`` works end-to-end."""
        env_data = make_app_config_env()
        patched_cmd_envs.get_env.return_value = env_data
        patched_list_models.return_value = ["m1"]
        with patched_resolve_password_fallback("envpass"):
            resp = await client.get("/api/envs/default/models")

        assert resp.status_code == 200
        patched_list_models.assert_awaited_once()
        _args, kwargs = patched_list_models.call_args
        assert kwargs.get("password") == "envpass"


class TestRevealEnvValue:
    """POST /api/envs/<name>/reveal — reveals the unredacted value of an
    env key. The decryption password comes from the ``yadc_password``
    session cookie (with the ``YADC_PASSWORD`` env-var fallback)."""

    @pytest.mark.asyncio
    async def test_cookie_password_decrypts_value(self, client, patched_cmd_envs, cookie_password):
        """When the cookie is set, the value is forwarded to
        ``cmd_envs.decrypt_setting`` and the decrypted text is returned."""
        env_data = make_app_config_env(
            api_token="encrypted-token",
            encryption_method="password",
        )
        patched_cmd_envs.get_env.return_value = env_data
        patched_cmd_envs.decrypt_setting = MagicMock(return_value="decrypted-secret")
        cookie_password(client, "hunter2")

        resp = await client.post("/api/envs/default/reveal", json={"key": "api_token"})

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == {"value": "decrypted-secret"}
        patched_cmd_envs.decrypt_setting.assert_called_once()
        _args, kwargs = patched_cmd_envs.decrypt_setting.call_args
        assert _args[0] == "encrypted-token"
        assert kwargs.get("password") == "hunter2"

    @pytest.mark.asyncio
    async def test_403_when_no_cookie_and_password_encrypted(self, client, patched_cmd_envs):
        """No cookie + no env-var → ``PasswordRequiredError`` propagates
        from ``cmd_envs.decrypt_setting`` and surfaces as 403
        ``PASSWORD_REQUIRED``."""
        env_data = make_app_config_env(
            api_token="encrypted-token",
            encryption_method="password",
        )
        patched_cmd_envs.get_env.return_value = env_data
        # Restore the real exception class on the mocked module so the
        # controller's ``except cmd_envs.PasswordRequiredError`` clause
        # matches. (Without this, ``cmd_envs.PasswordRequiredError`` is
        # a MagicMock attribute and the ``except`` raises TypeError.)
        patched_cmd_envs.PasswordRequiredError = PasswordRequiredError
        patched_cmd_envs.decrypt_setting = MagicMock(side_effect=PasswordRequiredError("test"))

        resp = await client.post("/api/envs/default/reveal", json={"key": "api_token"})

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["code"] == "PASSWORD_REQUIRED"
        assert "Password required" in data["error"]

    @pytest.mark.asyncio
    async def test_falls_back_to_yadc_password_env(self, client, patched_cmd_envs, patched_resolve_password_fallback):
        """No cookie + ``YADC_PASSWORD`` env var set → the env-var value
        is forwarded to ``cmd_envs.decrypt_setting``."""
        env_data = make_app_config_env(
            api_token="encrypted-token",
            encryption_method="password",
        )
        patched_cmd_envs.get_env.return_value = env_data
        patched_cmd_envs.decrypt_setting = MagicMock(return_value="decrypted-secret")

        with patched_resolve_password_fallback("envpass"):
            resp = await client.post("/api/envs/default/reveal", json={"key": "api_token"})

        assert resp.status_code == 200
        patched_cmd_envs.decrypt_setting.assert_called_once()
        _args, kwargs = patched_cmd_envs.decrypt_setting.call_args
        assert kwargs.get("password") == "envpass"

    @pytest.mark.asyncio
    async def test_plaintext_value_returned_without_password(self, client, patched_cmd_envs):
        """Non-encrypted values don't need a password — the cookie is
        ignored and the plaintext is returned as-is."""
        env_data = make_app_config_env(api_token="plaintext-token", encryption_method="none")
        patched_cmd_envs.get_env.return_value = env_data

        resp = await client.post("/api/envs/default/reveal", json={"key": "api_token"})

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data == {"value": "plaintext-token"}
        patched_cmd_envs.decrypt_setting.assert_not_called()
