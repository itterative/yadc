import re

import mock
import pytest

from tests.captioners.api.conftest import make_captioner
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


def _gemini_model(name: str, *, thinking: bool = True) -> dict:
    return {
        "name": name,
        "version": "1",
        "displayName": "Model",
        "supportedGenerationMethods": ["generateContent"],
        "thinking": thinking,
    }


@pytest.fixture
def gemini(load_test_data):
    def _gemini(case: str, model: str, base_url: str = "mock://generativelanguage.googleapis.com/v1beta"):
        from tests.captioners.api.conftest import MockAsyncSession

        session = MockAsyncSession()
        session.register_uri("GET", "models", json={"models": [_gemini_model(model)]})
        session.register_uri("GET", f"models/{model}", json=_gemini_model(model))
        session.register_uri("GET", "models/unknown", status_code=404, text="not found")
        session.register_uri("POST", f"models/{model}:generateContent", text=load_test_data(case))
        session.register_uri("POST", f"models/{model}:streamGenerateContent?alt=sse", text=load_test_data(case))

        captioner, _ = make_captioner(APITypes.GEMINI, api_url=base_url, api_token="secret token", session=session)
        return captioner

    return _gemini


class TestGemini:
    """Gemini captioner — basic prediction, streaming, and error paths.

    Note: ``predict()`` and ``predict_stream()`` share the client's streaming
    code path (the client is stream-only; ``predict`` just collects), so both
    point at the SSE fixtures.
    """

    @pytest.mark.asyncio
    async def test_predict(self, gemini, load_test_data):
        captioner: APICaptioner = gemini("streaming/gemini.txt", "gemini-2.5-flash")
        await captioner.load_model("gemini-2.5-flash")

        expected = load_test_data("streaming/gemini_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_streaming(self, gemini, load_test_data):
        captioner: APICaptioner = gemini("streaming/gemini.txt", "gemini-2.5-flash")
        await captioner.load_model("gemini-2.5-flash")

        expected = load_test_data("streaming/gemini_result.txt")
        got = "".join([token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))])

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_raises_error_on_bad_model(self, gemini):
        captioner: APICaptioner = gemini("streaming/gemini.txt", "gemini-2.5-flash")

        with pytest.raises(ValueError, match=re.compile("model not found: .*")):
            await captioner.load_model("unknown")

    @pytest.mark.asyncio
    async def test_thinking_flag_captured_via_discovery_path(self):
        """When the direct ``GET models/<name>`` returns 404, the client's
        ``_load_model`` falls back to the paginated discovery loop. The matched
        model's ``thinking`` flag must still be captured — it gates
        ``thinkingConfig`` in the request body.
        """
        from tests.captioners.api.conftest import MockAsyncSession

        session = MockAsyncSession()
        # Direct fetch fails — forces the discovery path.
        session.register_uri("GET", "models/gemini-2.5-flash", status_code=404, text="not found")
        session.register_uri("GET", "models", json={"models": [_gemini_model("models/gemini-2.5-flash", thinking=True)]})

        captioner, _ = make_captioner(APITypes.GEMINI, api_url="mock://generativelanguage.googleapis.com/v1beta", api_token="secret token", session=session)
        await captioner.load_model("gemini-2.5-flash")

        client = captioner.client
        assert client.model == "gemini-2.5-flash", "model not set from discovery path"
        assert client._is_thinking_model is True, "thinking flag must be captured from the matched GeminiModel"

    @pytest.mark.asyncio
    async def test_thinking_flag_false_via_discovery_path(self):
        """Same as above, but for a non-thinking model — the flag must be False."""
        from tests.captioners.api.conftest import MockAsyncSession

        session = MockAsyncSession()
        session.register_uri("GET", "models/gemini-2.0-flash", status_code=404, text="not found")
        session.register_uri("GET", "models", json={"models": [_gemini_model("models/gemini-2.0-flash", thinking=False)]})

        captioner, _ = make_captioner(APITypes.GEMINI, api_url="mock://generativelanguage.googleapis.com/v1beta", api_token="secret token", session=session)
        await captioner.load_model("gemini-2.0-flash")

        client = captioner.client
        assert client.model == "gemini-2.0-flash"
        assert client._is_thinking_model is False
