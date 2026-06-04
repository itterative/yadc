import re

import mock
import pytest

from tests.captioners.api.conftest import MockAsyncSession
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


@pytest.fixture
def gemini(load_test_data):
    def _gemini(case: str, model: str, base_url: str = "mock://generativelanguage.googleapis.com/v1beta"):
        session = MockAsyncSession()
        session.register_uri(
            "GET",
            "models",
            json={
                "models": [
                    {
                        "name": model,
                        "version": "1",
                        "displayName": "Model",
                        "supportedGenerationMethods": ["generateContent"],
                        "thinking": True,
                    }
                ]
            },
        )
        session.register_uri(
            "GET",
            f"models/{model}",
            json={
                "name": model,
                "version": "1",
                "displayName": "Model",
                "supportedGenerationMethods": ["generateContent"],
                "thinking": True,
            },
        )
        session.register_uri("GET", "models/unknown", status_code=404, text="not found")
        session.register_uri("POST", f"models/{model}:generateContent", text=load_test_data(case))
        session.register_uri("POST", f"models/{model}:streamGenerateContent?alt=sse", text=load_test_data(case))

        captioner = APICaptioner(
            api_type=APITypes.GEMINI,
            api_url=base_url,
            api_token="secret token",
            async_session=session,
        )

        return captioner

    return _gemini


class TestGemini:
    """Gemini captioner — basic prediction, streaming, and error paths."""

    @pytest.mark.asyncio
    async def test_predict(self, gemini, load_test_data):
        captioner: APICaptioner = gemini("nonstreaming/gemini.txt", "gemini-2.5-flash")
        await captioner.load_model("gemini-2.5-flash")

        expected = load_test_data("nonstreaming/gemini_result.txt")
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
        captioner: APICaptioner = gemini("nonstreaming/gemini.txt", "gemini-2.5-flash")

        with pytest.raises(ValueError, match=re.compile("model not found: .*")):
            await captioner.load_model("unknown")
