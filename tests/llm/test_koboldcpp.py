"""``KoboldcppLLMClient`` — bespoke ``load_model`` (reload + poll) and body customization."""

import json

import pytest

from yadc.llm import Message
from yadc.llm.koboldcpp import KoboldcppLLMClient


def _make_client(session, *, url="http://localhost:5001/v1"):
    return KoboldcppLLMClient(api_url=url, async_session=session)


class TestLoadModel:
    @pytest.mark.asyncio
    async def test_already_loaded_skips_reload(self, make_session):
        session = make_session()
        session.register_uri("GET", "/api/v1/model", json={"result": "koboldcpp/gemma"})
        session.register_uri("GET", "/api/admin/list_options", json=[])  # must not be reached

        client = _make_client(session)
        await client.load_model("gemma")

        assert client.model == "koboldcpp/gemma"

    @pytest.mark.asyncio
    async def test_reload_and_poll_until_active(self, make_session):
        calls = {"current": []}

        def _current(method, path, **kwargs):
            # First poll (after reload) returns inactive, then active.
            result = "inactive" if len(calls["current"]) == 0 else "koboldcpp/newmodel"
            calls["current"].append(result)
            return json.dumps({"result": result})

        session = make_session()
        session.register_uri("GET", "/api/v1/model", text=_current)
        session.register_uri("GET", "/api/admin/list_options", json=["newmodel.kcpps", "unload_model"])
        session.register_uri("POST", "/api/admin/reload_config", json={"success": True})

        client = _make_client(session)
        await client.load_model("newmodel")

        assert client.model == "koboldcpp/newmodel"

    @pytest.mark.asyncio
    async def test_not_found_lists_available(self, make_session):
        session = make_session()
        session.register_uri("GET", "/api/v1/model", json={"result": "other"})
        session.register_uri("GET", "/api/admin/list_options", json=["a.kcpps", "b.kcpps"])

        client = _make_client(session)
        with pytest.raises(ValueError, match="model not found: missing.*a.kcpps.*b.kcpps"):
            await client.load_model("missing")

    @pytest.mark.asyncio
    async def test_reload_failure_raises(self, make_session):
        session = make_session()
        session.register_uri("GET", "/api/v1/model", json={"result": "other"})
        session.register_uri("GET", "/api/admin/list_options", json=["m.kcpps"])
        session.register_uri("POST", "/api/admin/reload_config", json={"success": False})

        client = _make_client(session)
        with pytest.raises(ValueError, match="failed to load model: m"):
            await client.load_model("m")

    @pytest.mark.asyncio
    async def test_list_models_filters_unload(self, make_session):
        session = make_session()
        session.register_uri("GET", "/api/admin/list_options", json=["a.kcpps", "unload_model", "b.kcpps"])

        client = _make_client(session)
        assert await client.list_models() == ["a.kcpps", "b.kcpps"]


class TestBody:
    @pytest.mark.asyncio
    async def test_stop_token_and_max_tokens(self, make_session):
        captured: dict = {}

        def _capture(method, path, **kwargs):
            captured.update(kwargs.get("json", {}))
            return "data: [DONE]\n\n"

        session = make_session()
        session.register_uri("POST", "chat/completions", text=_capture)

        client = _make_client(session)
        client._model = "m"

        await client.predict_next_message_stream([Message(role="user", content="hi")], max_tokens=64)

        assert captured["max_tokens"] == 64
        assert "max_completion_tokens" not in captured
        assert captured["stop"] == ["<|im_end|>"]

    @pytest.mark.asyncio
    async def test_streams_real_payload(self, make_session, load_test_data):
        session = make_session()
        session.register_uri("GET", "/api/v1/model", json={"result": "koboldcpp/m"})
        session.register_uri("POST", "chat/completions", text=load_test_data("streaming/koboldcpp.txt"))

        client = _make_client(session)
        await client.load_model("m")

        stream = await client.predict_next_message_stream([Message(role="user", content="hi")])
        message = await stream.collect()

        expected = load_test_data("streaming/koboldcpp_result.txt")
        assert message.content.strip() == expected.strip()
