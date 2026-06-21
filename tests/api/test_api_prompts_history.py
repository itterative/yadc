"""Tests for the ``/api/prompts/history/...`` endpoints.

Controller-level only — the :class:`PromptHistoryService` is mocked
so we exercise the HTTP surface (status codes, response shape,
body validation, the 404 paths) without hitting the real DB.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from quart import Quart

from yadc.api.configuration import Configuration
from yadc.api.controllers.api_prompts_history import api_prompts_history
from yadc.api.controllers.blueprints import ApiBlueprint
from yadc.api.services.prompt_history import PromptHistoryPage


def _user_config() -> Configuration:
    """Default test configuration — not used by the service, but the test fixture needs one."""
    return Configuration()


@pytest.fixture
def prompt_history_mock() -> MagicMock:
    """Mocked ``PromptHistoryService`` with sensible defaults."""
    mock = MagicMock()
    mock.save_entry.return_value = 1
    mock.get_entry.return_value = {
        "id": 1,
        "mode": "generate",
        "intent": "x",
        "focus": "both",
        "examples": [],
        "template_content": None,
        "created_t": 1234567890.0,
    }
    mock.delete_entry.return_value = True
    page = PromptHistoryPage(entries=[], next_token=None)
    mock.list_history.return_value = page
    return mock


@pytest.fixture
def client(test_configuration: Configuration, prompt_history_mock: MagicMock):
    """Quart test client with the history blueprint registered."""
    app = Quart(__name__)
    bp = ApiBlueprint("api", __name__, url_prefix="/api")

    api_prompts_history(bp, prompt_history_mock)

    app.register_blueprint(bp)

    client = app.test_client()
    client._prompt_history = prompt_history_mock  # type: ignore[attr-defined]
    return client


# --- POST /api/prompts/history ---


class TestSaveHistory:
    @pytest.mark.asyncio
    async def test_save_returns_full_entry(self, client, prompt_history_mock):
        """A successful save returns the full entry (with the new id)."""
        resp = await client.post(
            "/api/prompts/history",
            json={
                "mode": "generate",
                "intent": "Caption each image as a single sentence.",
                "focus": "both",
                "examples": [],
            },
        )

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["id"] == 1
        assert data["intent"] == "x"
        # The mock's get_entry is what the controller uses to read back.
        prompt_history_mock.get_entry.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_save_validates_body(self, client):
        """Body validation runs first — an invalid body → 400, no service call."""
        resp = await client.post(
            "/api/prompts/history",
            json={"mode": "bogus", "intent": "x", "focus": "both"},
        )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_save_rejects_extra_field(self, client):
        """``extra='forbid'`` on the body model → unknown fields rejected."""
        resp = await client.post(
            "/api/prompts/history",
            json={
                "mode": "generate",
                "intent": "x",
                "focus": "both",
                "unknown_field": "y",
            },
        )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_save_passes_template_content_for_refine(self, client, prompt_history_mock):
        """A refine-mode save forwards the ``template_content`` to the service."""
        resp = await client.post(
            "/api/prompts/history",
            json={
                "mode": "refine",
                "intent": "refine this",
                "focus": "system",
                "template_content": "{% set system_prompt %}\nfoo\n{% endset %}",
            },
        )

        assert resp.status_code == 200
        save_call = prompt_history_mock.save_entry.call_args
        req = save_call.args[0]
        assert req.mode == "refine"
        assert req.template_content == "{% set system_prompt %}\nfoo\n{% endset %}"

    @pytest.mark.asyncio
    async def test_save_500_if_read_back_fails(self, client, prompt_history_mock):
        """If the post-insert read-back returns ``None`` (shouldn't happen) → 500."""
        prompt_history_mock.get_entry.return_value = None

        resp = await client.post(
            "/api/prompts/history",
            json={"mode": "generate", "intent": "x", "focus": "both"},
        )

        assert resp.status_code == 500


# --- GET /api/prompts/history ---


class TestListHistory:
    @pytest.mark.asyncio
    async def test_list_returns_page(self, client, prompt_history_mock):
        """The list endpoint returns the page as a JSON dict."""
        prompt_history_mock.list_history.return_value = PromptHistoryPage(
            entries=[
                {
                    "id": 2,
                    "mode": "refine",
                    "had_template": True,
                    "focus": "system",
                    "intent_preview": "refine this",
                    "example_count": 0,
                    "created_t": 1234567890.0,
                }
            ],
            next_token=None,
        )

        resp = await client.get("/api/prompts/history")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["had_template"] is True
        assert data["next_token"] is None

    @pytest.mark.asyncio
    async def test_list_passes_limit_and_next(self, client, prompt_history_mock):
        """The ``limit`` and ``next`` query params are forwarded to the service."""
        prompt_history_mock.list_history.return_value = PromptHistoryPage(entries=[], next_token=None)

        await client.get("/api/prompts/history?limit=10&next=42")

        prompt_history_mock.list_history.assert_called_once_with(limit=10, next_token="42")

    @pytest.mark.asyncio
    async def test_list_default_limit(self, client, prompt_history_mock):
        """No ``limit`` query param → default of 50."""
        prompt_history_mock.list_history.return_value = PromptHistoryPage(entries=[], next_token=None)

        await client.get("/api/prompts/history")

        _, kwargs = prompt_history_mock.list_history.call_args
        assert kwargs["limit"] == 50


# --- GET /api/prompts/history/<id> ---


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_get_returns_full_entry(self, client, prompt_history_mock):
        prompt_history_mock.get_entry.return_value = {
            "id": 42,
            "mode": "refine",
            "intent": "refine this",
            "focus": "system",
            "examples": [
                {"subject": "a", "caption": "b", "image_data_url": "data:image/png;base64,AAA"},
            ],
            "template_content": "{% set system_prompt %}\nfoo\n{% endset %}",
            "created_t": 1234567890.0,
        }

        resp = await client.get("/api/prompts/history/42")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["id"] == 42
        assert data["mode"] == "refine"
        assert len(data["examples"]) == 1

    @pytest.mark.asyncio
    async def test_get_returns_404_for_missing(self, client, prompt_history_mock):
        prompt_history_mock.get_entry.return_value = None

        resp = await client.get("/api/prompts/history/9999")

        assert resp.status_code == 404
        data = await resp.get_json()
        assert data["code"] == "NOT_FOUND"


# --- DELETE /api/prompts/history/<id> ---


class TestDeleteHistory:
    @pytest.mark.asyncio
    async def test_delete_returns_ok(self, client, prompt_history_mock):
        prompt_history_mock.delete_entry.return_value = True

        resp = await client.delete("/api/prompts/history/42")

        assert resp.status_code == 200
        data = await resp.get_json()
        assert data["status"] == "ok"
        prompt_history_mock.delete_entry.assert_called_once_with(42)

    @pytest.mark.asyncio
    async def test_delete_returns_404_for_missing(self, client, prompt_history_mock):
        prompt_history_mock.delete_entry.return_value = False

        resp = await client.delete("/api/prompts/history/9999")

        assert resp.status_code == 404
        data = await resp.get_json()
        assert data["code"] == "NOT_FOUND"
