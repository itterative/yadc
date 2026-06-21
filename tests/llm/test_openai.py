"""``OpenAICompatibleLLMClient`` / ``OpenAILLMClient`` — streaming, reasoning, errors, models."""

import re

import httpx
import pytest

from yadc.llm import Message, OpenAILLMClient, OpenRouterLLMClient
from yadc.llm.llamacpp import LlamacppLLMClient


def _openai_chunk(delta: dict, *, finish_reason=None, object_id="1", usage=None, error=None) -> str:
    import json

    payload = {
        "id": object_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    if error is not None:
        payload["error"] = error
    return "data: " + json.dumps(payload)


def _sse(*lines: str, done: bool = True) -> str:
    body = "".join(line + "\n\n" for line in lines)
    if done:
        body += "data: [DONE]\n\n"
    return body


def _make_client(cls, session, *, url="https://api.openai.com/v1", token="t"):
    return cls(api_url=url, api_token=token, async_session=session)


class TestStreaming:
    """Core streaming + accumulation against real captured payloads."""

    @pytest.mark.asyncio
    async def test_collect_accumulates_content(self, make_session, load_test_data):
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "o4-mini", "object": "model", "owned_by": "openai"}]})
        session.register_uri("POST", "chat/completions", text=load_test_data("streaming/openai_o4_mini.txt"))

        client = _make_client(OpenAILLMClient, session)
        await client.load_model("o4-mini")

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        expected = load_test_data("streaming/openai_o4_mini_result.txt")
        assert message.content.strip() == expected.strip()
        assert message.role == "assistant"

    @pytest.mark.asyncio
    async def test_iteration_yields_chunks_and_updates_message(self, make_session):
        session = make_session()
        sse = _sse(
            _openai_chunk({"role": "assistant", "content": ""}),
            _openai_chunk({"content": "Hel"}),
            _openai_chunk({"content": "lo"}),
            _openai_chunk({}, finish_reason="stop"),
        )
        session.register_uri("POST", "chat/completions", text=sse)

        client = _make_client(OpenAILLMClient, session)
        client._model = "m"  # bypass load_model for a focused decoder test

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])

        texts = []
        async for chunk in stream:
            if chunk.text:
                texts.append(chunk.text)
            # .message reflects every chunk yielded so far
            assert stream.message.content == "".join(texts)

        assert texts == ["Hel", "lo"]
        assert stream.message.content == "Hello"

    @pytest.mark.asyncio
    async def test_reasoning_separated_from_content(self, make_session):
        session = make_session()
        sse = _sse(
            _openai_chunk({"reasoning_content": "thinking..."}),
            _openai_chunk({"reasoning": " more"}),
            _openai_chunk({"content": "answer"}),
            _openai_chunk({}, finish_reason="stop"),
        )
        session.register_uri("POST", "chat/completions", text=sse)

        client = _make_client(OpenAILLMClient, session)
        client._model = "m"

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        assert message.content == "answer"
        assert message.reasoning == "thinking... more"

    @pytest.mark.asyncio
    async def test_reasoning_encrypted_captured(self, make_session):
        session = make_session()
        detail = {"type": "reasoning.encrypted", "data": "opaque"}
        sse = _sse(
            _openai_chunk({"content": "answer"}),
            _openai_chunk({"reasoning_details": [detail]}),
            _openai_chunk({}, finish_reason="stop"),
        )
        session.register_uri("POST", "chat/completions", text=sse)

        client = _make_client(OpenAILLMClient, session)
        client._model = "m"

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        assert message.content == "answer"
        assert message.reasoning_encrypted == [detail]


class TestErrors:
    @pytest.mark.asyncio
    async def test_http_status_error_normalized(self, make_session):
        session = make_session()
        session.register_uri(
            "POST",
            "chat/completions",
            status_code=401,
            text='{"error": {"code": 401, "message": "bad key"}}',
        )

        client = _make_client(OpenAILLMClient, session)
        client._model = "m"

        with pytest.raises(ValueError, match="api returned an error"):
            await client.predict_next_message_stream([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_connection_closed_wrapped(self, make_session):
        session = make_session()
        session.register_uri("POST", "chat/completions", exc=httpx.RemoteProtocolError("peer gone"))

        client = _make_client(OpenAILLMClient, session)
        client._model = "m"

        with pytest.raises(ValueError, match="Connection closed unexpectedly"):
            await (await client.predict_next_message_stream([Message(role="user", content="hi")])).collect()

    @pytest.mark.asyncio
    async def test_no_model_raises(self, make_session):
        client = _make_client(OpenAILLMClient, make_session())
        with pytest.raises(ValueError, match="no model loaded"):
            await client.predict_next_message_stream([Message(role="user", content="hi")])


class TestModels:
    @pytest.mark.asyncio
    async def test_list_models(self, make_session):
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "a"}, {"id": "b"}]})

        client = _make_client(OpenAILLMClient, session)
        assert await client.list_models() == ["a", "b"]

    @pytest.mark.asyncio
    async def test_load_model_existence_check(self, make_session):
        session = make_session()
        session.register_uri("GET", "models", json={"data": [{"id": "a"}, {"id": "b"}]})

        client = _make_client(OpenAILLMClient, session)
        await client.load_model("a")
        assert client.model == "a"

        with pytest.raises(ValueError, match=re.compile("model not found: unknown")):
            await client.load_model("unknown")


class TestCustomizeBody:
    """Per-server body tweaks flow through to the request body."""

    @pytest.mark.asyncio
    async def test_llamacpp_uses_max_tokens(self, make_session):
        captured: dict = {}

        def _capture(method, path, **kwargs):
            captured.update(kwargs.get("json", {}))
            return '{"data":[]}'

        session = make_session()
        session.register_uri("POST", "chat/completions", text=_capture)
        session.register_uri("GET", "models", text='{"data":[]}')

        client = _make_client(LlamacppLLMClient, session, url="http://localhost:8080/v1")
        client._model = "m"

        await client.predict_next_message_stream([Message(role="user", content="hi")], max_tokens=128)
        # llama.cpp expects max_tokens, not max_completion_tokens
        assert captured.get("max_tokens") == 128
        assert "max_completion_tokens" not in captured

    @pytest.mark.asyncio
    async def test_openrouter_reasoning_block(self, make_session):
        captured: dict = {}

        def _capture(method, path, **kwargs):
            captured.update(kwargs.get("json", {}))
            return ""

        session = make_session()
        session.register_uri("POST", "chat/completions", text=_capture)

        client = _make_client(OpenRouterLLMClient, session, url="https://openrouter.ai/api/v1")
        client._model = "m"

        await client.predict_next_message_stream([Message(role="user", content="hi")], reasoning="high")
        assert captured["reasoning"] == {"enabled": True, "effort": "high", "exclude": False}
        assert "reasoning_effort" not in captured
        assert captured["max_tokens"] == 4096
        assert captured["usage"] == {"include": True}
        assert "_reasoning_exclude" not in captured

    @pytest.mark.asyncio
    async def test_openrouter_reasoning_redacted_not_surfaceced(self, make_session):
        session = make_session()
        sse = _sse(
            _openai_chunk({"content": "answer"}),
            _openai_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "[REDACTED]"}]}),
            _openai_chunk({}, finish_reason="stop"),
        )
        session.register_uri("POST", "chat/completions", text=sse)

        client = _make_client(OpenRouterLLMClient, session, url="https://openrouter.ai/api/v1")
        client._model = "m"

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        assert message.content == "answer"
        assert message.reasoning is None  # redacted reasoning is dropped
