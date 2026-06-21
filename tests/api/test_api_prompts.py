"""Tests for the ``/api/prompts/generate`` endpoint.

Controller-level only — the :class:`PromptGenerationService` is mocked
so we exercise the HTTP surface (status codes, NDJSON stream shape,
password + misconfig preflight, body validation) without hitting any
real LLM backend.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from quart import Quart

from yadc.api.configuration import Configuration
from yadc.api.controllers.api_prompts import api_prompts
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.cmd.envs.keystorage_password import PasswordRequiredError
from yadc.core.user_config import UserConfig, UserConfigApi
from yadc.llm import StreamChunk

_PATCH_CMD_ENVS = "yadc.api.controllers.api_prompts.cmd_envs"


def _user_config(url: str = "https://api.example.com/v1", token: str = "tk", model: str = "gpt-x") -> UserConfig:
    return UserConfig(api=UserConfigApi(url=url, token=token, model_name=model))


def _json_lines(body: bytes) -> list[dict[str, Any]]:
    """Parse an NDJSON body into a list of dicts (one per line)."""
    return [json.loads(line) for line in body.splitlines() if line]


def _make_prompt_service(chunks: list[StreamChunk] | None = None) -> MagicMock:
    """Build a mocked ``PromptGenerationService`` whose ``generate`` yields ``chunks``.

    ``generate`` is a :class:`MagicMock` (so tests can introspect call args)
    whose ``side_effect`` is an async generator. Tests that need different
    behaviour reassign ``service.generate`` to a fresh mock.
    """
    service = MagicMock()
    chunks = chunks or []

    def _gen(*_args: Any, **_kwargs: Any) -> AsyncIterator[StreamChunk]:
        async def _iter() -> AsyncIterator[StreamChunk]:
            for c in chunks:
                yield c

        return _iter()

    service.generate = MagicMock(side_effect=_gen)

    return service


@pytest.fixture
def client(test_configuration: Configuration):
    """Quart test client with the prompts blueprint registered."""
    app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    mock_logging = MagicMock()
    mock_logging.get_logger.return_value = MagicMock()

    prompt_service = _make_prompt_service([StreamChunk(text="ok")])

    api_prompts(bp, mock_logging, prompt_service)

    app.register_blueprint(bp)

    client = app.test_client()
    client._prompt_service = prompt_service  # type: ignore[attr-defined]
    return client


@pytest.fixture
def patched_cmd_envs():
    """Patch the ``cmd_envs`` module in the controller."""
    with patch(_PATCH_CMD_ENVS) as mock_cmd_envs:
        yield mock_cmd_envs


@pytest.fixture
def cookie_password():
    """Factory fixture: set the ``yadc_password`` cookie on the test client."""

    def _set(client, value):
        client.set_cookie("localhost", "yadc_password", value, path="/api")

    return _set


class TestGeneratePreflight:
    """Preflight checks — must surface the right HTTP status code BEFORE the stream starts."""

    @pytest.mark.asyncio
    async def test_password_required_returns_403(self, client, patched_cmd_envs):
        """``PasswordRequiredError`` from ``cmd_envs.load_env`` → 403 ``PASSWORD_REQUIRED``."""
        patched_cmd_envs.load_env.side_effect = PasswordRequiredError("locked")

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 403
        data = await resp.get_json()
        assert data["code"] == "PASSWORD_REQUIRED"

    @pytest.mark.asyncio
    async def test_missing_api_url_returns_400(self, client, patched_cmd_envs):
        """Env with no ``api_url`` → 400 ``BAD_REQUEST`` (not a stream-level error)."""
        patched_cmd_envs.load_env.return_value = _user_config(url="")

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        assert "api_url" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_model_name_returns_400(self, client, patched_cmd_envs):
        """Env with no ``api_model_name`` and no override → 400 ``BAD_REQUEST``."""
        patched_cmd_envs.load_env.return_value = _user_config(model="")

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"
        assert "api_model_name" in data["error"]

    @pytest.mark.asyncio
    async def test_api_model_name_override_accepted(self, client, patched_cmd_envs):
        """A body ``api_model_name`` overrides the env's empty default → preflight passes."""
        patched_cmd_envs.load_env.return_value = _user_config(model="")
        client._prompt_service.generate = _async_iter_mock([StreamChunk(text="ok")])  # type: ignore[attr-defined]

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x", "api_model_name": "override"})

        assert resp.status_code == 200
        # Service was called with the overridden model.
        client._prompt_service.generate.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_invalid_body_returns_400(self, client):
        """Body validation runs BEFORE preflight — invalid body → 400, no env decryption."""
        resp = await client.post("/api/prompts/generate", json={"env": "default"})  # no intent

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["code"] == "BAD_REQUEST"

    @pytest.mark.asyncio
    async def test_extra_field_returns_400(self, client):
        """``model_config.extra = 'forbid'`` → unknown fields rejected."""
        resp = await client.post(
            "/api/prompts/generate",
            json={"env": "default", "intent": "x", "unknown_field": "y"},
        )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_focus_returns_400(self, client):
        """``focus`` is a Literal — invalid values rejected at validation."""
        resp = await client.post(
            "/api/prompts/generate",
            json={"env": "default", "intent": "x", "focus": "everywhere"},
        )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_json_body_returns_400(self, client):
        """No JSON body → 400 (silent JSON parsing failure surfaces as validation error)."""
        resp = await client.post("/api/prompts/generate", data="not json")

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_example_missing_image_data_url_returns_400(self, client):
        """``ExamplePair.image_data_url`` is required — an example without it → 400."""
        resp = await client.post(
            "/api/prompts/generate",
            json={
                "env": "default",
                "intent": "caption cats",
                "examples": [{"subject": "a cat", "caption": "a tabby"}],
            },
        )

        assert resp.status_code == 400


class TestGenerateStreaming:
    """Streaming behaviour — NDJSON shape, chunk translation, ``done`` terminator."""

    @pytest.mark.asyncio
    async def test_streams_text_chunks_as_token_lines(self, client, patched_cmd_envs):
        """Each ``StreamChunk`` with ``text`` → one NDJSON line ``{"type": "token", "text": "..."}``."""
        chunks = [
            StreamChunk(text="{% set "),
            StreamChunk(text="system_prompt %}"),
            StreamChunk(text="hi"),
        ]
        client._prompt_service.generate = _async_iter_mock(chunks)  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 200
        assert resp.mimetype == "application/x-ndjson"
        body = await resp.get_data()
        lines = _json_lines(body)
        assert lines == [
            {"type": "token", "text": "{% set "},
            {"type": "token", "text": "system_prompt %}"},
            {"type": "token", "text": "hi"},
            {"type": "done"},
        ]

    @pytest.mark.asyncio
    async def test_streams_reasoning_chunks(self, client, patched_cmd_envs):
        """``reasoning`` and ``reasoning_summary`` flow through to the NDJSON line."""
        chunks = [
            StreamChunk(reasoning="thinking..."),
            StreamChunk(text="answer"),
        ]
        client._prompt_service.generate = _async_iter_mock(chunks)  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        lines = _json_lines(await resp.get_data())
        assert lines[0] == {"type": "token", "reasoning": "thinking..."}
        assert lines[1] == {"type": "token", "text": "answer"}
        assert lines[-1] == {"type": "done"}

    @pytest.mark.asyncio
    async def test_empty_stream_emits_error_line(self, client, patched_cmd_envs):
        """A 200-OK stream that produces no text is treated as a failure.

        A model can finish without emitting content (safety refusal,
        reasoning-only response, network trim) — that's not a usable
        template, so we surface it as an error line instead of ``done``.
        """
        client._prompt_service.generate = _async_iter_mock([])  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 200
        lines = _json_lines(await resp.get_data())
        assert lines == [{"type": "error", "message": "Model returned an empty response"}]

    @pytest.mark.asyncio
    async def test_mid_stream_value_error_emits_error_line(self, client, patched_cmd_envs):
        """``ValueError`` from the service mid-stream → ``{"type": "error", ...}`` line (no crash)."""
        client._prompt_service.generate = _async_raising_mock(ValueError("upstream failed"))  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 200  # status was already 200 by the time we hit the stream
        lines = _json_lines(await resp.get_data())
        assert len(lines) == 1
        assert lines[0] == {"type": "error", "message": "upstream failed"}

    @pytest.mark.asyncio
    async def test_mid_stream_password_required_emits_error_line(self, client, patched_cmd_envs):
        """Defensive: if password-required surfaces mid-stream, emit an error line, don't crash."""
        client._prompt_service.generate = _async_raising_mock(PasswordRequiredError("locked"))  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        lines = _json_lines(await resp.get_data())
        assert len(lines) == 1
        assert lines[0]["type"] == "error"
        assert "Password required" in lines[0]["message"]

    @pytest.mark.asyncio
    async def test_mid_stream_unexpected_exception_emits_error_line(self, client, patched_cmd_envs):
        """An unexpected ``Exception`` mid-stream → generic ``error`` line (no 500)."""
        client._prompt_service.generate = _async_raising_mock(RuntimeError("kaboom"))  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 200
        lines = _json_lines(await resp.get_data())
        assert len(lines) == 1
        assert lines[0]["type"] == "error"
        assert "kaboom" in lines[0]["message"]

    @pytest.mark.asyncio
    async def test_response_has_no_cache_headers(self, client, patched_cmd_envs):
        """Streaming responses disable caching + buffering."""
        client._prompt_service.generate = _async_iter_mock([StreamChunk(text="x")])  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.headers.get("Cache-Control") == "no-cache"
        assert resp.headers.get("X-Accel-Buffering") == "no"


class TestGeneratePassesThrough:
    """The controller forwards body fields to the service correctly."""

    @pytest.mark.asyncio
    async def test_intent_examples_focus_forwarded(self, client, patched_cmd_envs):
        """The full body lands in the service's ``request=...`` kwarg."""
        client._prompt_service.generate = _async_iter_mock([StreamChunk(text="x")])  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        # Tiny data URL — enough to satisfy Pydantic, the controller never decodes it.
        examples = [
            {"subject": "a", "caption": "b", "image_data_url": "data:image/png;base64,AAA"},
            {"subject": "c", "caption": "d", "image_data_url": "data:image/png;base64,BBB"},
        ]
        resp = await client.post(
            "/api/prompts/generate",
            json={"env": "my-env", "intent": "caption cats", "examples": examples, "focus": "user"},
        )

        assert resp.status_code == 200
        call = client._prompt_service.generate.call_args  # type: ignore[attr-defined]
        assert call.kwargs["password"] is None
        request = call.kwargs["request"]
        assert request.env == "my-env"
        assert request.intent == "caption cats"
        assert request.focus == "user"
        assert [ex.model_dump() for ex in request.examples] == examples

    @pytest.mark.asyncio
    async def test_password_resolved_from_cookie(self, client, patched_cmd_envs, cookie_password):
        """The ``yadc_password`` cookie is forwarded to the service as ``password``."""
        cookie_password(client, "secret")
        client._prompt_service.generate = _async_iter_mock([StreamChunk(text="x")])  # type: ignore[attr-defined]
        patched_cmd_envs.load_env.return_value = _user_config()

        resp = await client.post("/api/prompts/generate", json={"env": "default", "intent": "x"})

        assert resp.status_code == 200
        assert client._prompt_service.generate.call_args.kwargs["password"] == "secret"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_iter_mock(chunks: list[StreamChunk]) -> MagicMock:
    """Return a MagicMock whose ``side_effect`` yields the given chunks.

    The mock can be assigned directly to ``service.generate`` and inspected
    via ``call_args`` / ``assert_called_once_with`` like any other mock.
    """

    def _side_effect(*_args: Any, **_kwargs: Any) -> AsyncIterator[StreamChunk]:
        async def _iter() -> AsyncIterator[StreamChunk]:
            for c in chunks:
                yield c

        return _iter()

    return MagicMock(side_effect=_side_effect)


def _async_raising_mock(exc: BaseException) -> MagicMock:
    """Return a MagicMock whose ``side_effect`` raises ``exc`` mid-stream."""

    def _side_effect(*_args: Any, **_kwargs: Any) -> AsyncIterator[StreamChunk]:
        async def _iter() -> AsyncIterator[StreamChunk]:
            raise exc
            yield  # pragma: no cover

        return _iter()

    return MagicMock(side_effect=_side_effect)
