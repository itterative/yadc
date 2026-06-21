import re

import mock
import pytest

from tests.captioners.api.conftest import make_captioner
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


@pytest.fixture
def openrouter(load_test_data):
    def _openrouter(case: str, model: str, base_url: str = "mock://openrouter.ai/api/v1"):
        from tests.captioners.api.conftest import MockAsyncSession

        session = MockAsyncSession()
        session.register_uri("GET", "credits", json={"data": {"total_credits": 1.0, "total_usage": 0.1}})
        session.register_uri("GET", "models", json={"data": [{"id": model, "object": "model", "owned_by": "openrouter"}]})
        session.register_uri("POST", "chat/completions", text=load_test_data(case))

        captioner, _ = make_captioner(APITypes.OPENROUTER, api_url=base_url, session=session)
        return captioner

    return _openrouter


class TestOpenRouter:
    """OpenRouter captioner — basic prediction and streaming across multiple
    upstream models, plus error paths.

    Note: ``predict()`` and ``predict_stream()`` share the client's streaming
    code path (the client is stream-only; ``predict`` just collects), so both
    point at the SSE fixtures.
    """

    @pytest.mark.asyncio
    async def test_predict_gpt_5_mini(self, openrouter, load_test_data):
        captioner: APICaptioner = openrouter("streaming/openrouter_gpt_5_mini.txt", "openai/gpt-5-mini")
        await captioner.load_model("openai/gpt-5-mini")

        expected = load_test_data("streaming/openrouter_gpt_5_mini_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_streaming_gpt_5_mini(self, openrouter, load_test_data):
        captioner: APICaptioner = openrouter("streaming/openrouter_gpt_5_mini.txt", "openai/gpt-5-mini")
        await captioner.load_model("openai/gpt-5-mini")

        expected = load_test_data("streaming/openrouter_gpt_5_mini_result.txt")
        got = "".join([token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))])

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_predict_qwen3_vl(self, openrouter, load_test_data):
        captioner: APICaptioner = openrouter("streaming/openrouter_qwen3_vl.txt", "qwen/qwen3-vl-235b-a22b-thinking")
        await captioner.load_model("qwen/qwen3-vl-235b-a22b-thinking")

        expected = load_test_data("streaming/openrouter_qwen3_vl_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_streaming_qwen3_vl(self, openrouter, load_test_data):
        captioner: APICaptioner = openrouter("streaming/openrouter_qwen3_vl.txt", "qwen/qwen3-vl-235b-a22b-thinking")
        await captioner.load_model("qwen/qwen3-vl-235b-a22b-thinking")

        expected = load_test_data("streaming/openrouter_qwen3_vl_result.txt")
        got = "".join([token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))])

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_raises_error_on_bad_model(self, openrouter):
        captioner: APICaptioner = openrouter("streaming/openrouter_gpt_5_mini.txt", "openai/gpt-5-mini")

        with pytest.raises(ValueError, match=re.compile("model not found: .*")):
            await captioner.load_model("unknown")
