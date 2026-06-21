"""``GeminiLLMClient`` — streaming, reasoning separation, thinkingConfig, model loading."""

import json

import httpx
import pytest

from yadc.llm import Message
from yadc.llm.gemini import GeminiLLMClient


def _gemini_model(name: str, *, thinking: bool = True) -> dict:
    return {
        "name": name,
        "version": "1",
        "displayName": name,
        "supportedGenerationMethods": ["generateContent"],
        "thinking": thinking,
    }


def _make_client(session, *, url="https://generativelanguage.googleapis.com/v1beta", token="t"):
    return GeminiLLMClient(api_url=url, api_token=token, async_session=session)


def _gemini_sse(*objs: dict, done: bool = True) -> str:
    body = "".join("data: " + json.dumps(o) + "\n\n" for o in objs)
    if done:
        body += "data: [DONE]\n\n"
    return body


class TestStreaming:
    @pytest.mark.asyncio
    async def test_collect_against_real_payload(self, make_session, load_test_data):
        session = make_session()
        session.register_uri("GET", "models/gemini-2.5-flash", json=_gemini_model("models/gemini-2.5-flash"))
        session.register_uri(
            "POST",
            "models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            text=load_test_data("streaming/gemini.txt"),
        )

        client = _make_client(session)
        await client.load_model("gemini-2.5-flash")

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        expected = load_test_data("streaming/gemini_result.txt")
        assert message.content.strip() == expected.strip()
        # Gemini's `thought:true` parts land in reasoning, not content.
        assert message.reasoning and "Raccoon" in message.reasoning

    @pytest.mark.asyncio
    async def test_thought_parts_routed_to_reasoning(self, make_session):
        session = make_session()
        session.register_uri("GET", "models/m", json=_gemini_model("models/m"))
        session.register_uri(
            "POST",
            "models/m:streamGenerateContent?alt=sse",
            text=_gemini_sse(
                {"candidates": [{"content": {"parts": [{"text": "private musing", "thought": True}], "role": "model"}, "index": 0}]},
                {"candidates": [{"content": {"parts": [{"text": "answer"}], "role": "model"}, "index": 0}]},
                {
                    "candidates": [{"content": {"parts": [{"text": ""}], "role": "model"}, "index": 0}],
                    "usageMetadata": {"candidatesTokenCount": 1, "promptTokenCount": 2, "totalTokenCount": 3},
                },
            ),
        )

        client = _make_client(session)
        await client.load_model("m")

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        assert message.content == "answer"
        assert message.reasoning == "private musing"

    @pytest.mark.asyncio
    async def test_system_message_becomes_system_instruction(self, make_session):
        captured: dict = {}

        def _capture(method, path, **kwargs):
            captured.update(kwargs.get("json", {}))
            return _gemini_sse({"candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "index": 0}]})

        session = make_session()
        session.register_uri("GET", "models/m", json=_gemini_model("models/m"))
        session.register_uri("POST", "models/m:streamGenerateContent?alt=sse", text=_capture)

        client = _make_client(session)
        await client.load_model("m")

        await client.predict_next_message_stream([Message(role="system", content="be brief"), Message(role="user", content="hi")])

        assert captured["system_instruction"] == {"parts": [{"text": "be brief"}]}
        assert captured["contents"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_reasoning_enables_thinking_config(self, make_session):
        captured: dict = {}

        def _capture(method, path, **kwargs):
            captured.update(kwargs.get("json", {}))
            return _gemini_sse({"candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "index": 0}]})

        session = make_session()
        session.register_uri("GET", "models/m", json=_gemini_model("models/m", thinking=True))
        session.register_uri("POST", "models/m:streamGenerateContent?alt=sse", text=_capture)

        client = _make_client(session)
        await client.load_model("m")

        await client.predict_next_message_stream([Message(role="user", content="hi")], reasoning="high")

        assert captured["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 2048}

    @pytest.mark.asyncio
    async def test_connection_closed_wrapped(self, make_session):
        session = make_session()
        session.register_uri("GET", "models/m", json=_gemini_model("models/m"))
        session.register_uri("POST", "models/m:streamGenerateContent?alt=sse", exc=httpx.RemoteProtocolError("gone"))

        client = _make_client(session)
        await client.load_model("m")

        with pytest.raises(ValueError, match="Connection closed unexpectedly"):
            await (await client.predict_next_message_stream([Message(role="user", content="hi")])).collect()


class TestAuth:
    @pytest.mark.asyncio
    async def test_uses_x_goog_api_key_not_bearer(self, make_session):
        session = make_session()
        _make_client(session)
        assert session.headers.get("x-goog-api-key") == "t"
        assert "Authorization" not in session.headers

    @pytest.mark.asyncio
    async def test_requires_token(self, make_session):
        with pytest.raises(ValueError, match="no api_token"):
            GeminiLLMClient(api_url="https://x", api_token="", async_session=make_session())


class TestModelLoading:
    @pytest.mark.asyncio
    async def test_load_model_direct_fetch_captures_thinking_flag(self, make_session):
        session = make_session()
        session.register_uri("GET", "models/gemini-2.5-flash", json=_gemini_model("models/gemini-2.5-flash", thinking=True))

        client = _make_client(session)
        await client.load_model("gemini-2.5-flash")

        assert client.model == "gemini-2.5-flash"
        assert client._is_thinking_model is True

    @pytest.mark.asyncio
    async def test_load_model_discovery_path_on_404(self, make_session):
        session = make_session()
        session.register_uri("GET", "models/gemini-2.0-flash", status_code=404, text="not found")
        session.register_uri("GET", "models", json={"models": [_gemini_model("models/gemini-2.0-flash", thinking=False)]})

        client = _make_client(session)
        await client.load_model("gemini-2.0-flash")

        assert client.model == "gemini-2.0-flash"
        assert client._is_thinking_model is False

    @pytest.mark.asyncio
    async def test_load_model_not_found(self, make_session):
        session = make_session()
        session.register_uri("GET", "models/unknown", status_code=404, text="not found")
        session.register_uri("GET", "models", json={"models": [_gemini_model("models/gemini-2.5-flash")]})

        client = _make_client(session)
        with pytest.raises(ValueError, match="model not found: unknown"):
            await client.load_model("unknown")

    @pytest.mark.asyncio
    async def test_list_models_paginated_and_filtered(self, make_session):
        session = make_session()
        session.register_uri(
            "GET",
            "models",
            json={
                "models": [
                    _gemini_model("models/text-only", thinking=False) | {"supportedGenerationMethods": ["embedContent"]},
                    _gemini_model("models/gemini-2.5-flash"),
                ],
                "nextPageToken": "tok",
            },
        )
        session.register_uri(
            "GET",
            "models?pageToken=tok",
            json={"models": [_gemini_model("models/gemini-2.0-flash", thinking=False)], "nextPageToken": None},
        )

        client = _make_client(session)
        # text-only filtered out (no generateContent); both pages walked.
        assert await client.list_models() == ["gemini-2.5-flash", "gemini-2.0-flash"]
