import re

import mock
import pytest

from tests.captioners.api.conftest import make_captioner
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


@pytest.fixture
def llamacpp(load_test_data):
    def _llamacpp(case: str, model: str, base_url: str = "mock://llamacpp"):
        from tests.captioners.api.conftest import MockAsyncSession

        session = MockAsyncSession()
        session.register_uri("GET", "models", json={"data": [{"id": model, "object": "model", "owned_by": "llamacpp"}]})
        session.register_uri("POST", "chat/completions", text=load_test_data(case))

        captioner, _ = make_captioner(APITypes.LLAMACPP, api_url=f"{base_url}/v1", session=session)
        return captioner

    return _llamacpp


class TestLlamaCpp:
    """llama.cpp captioner — basic prediction, streaming, CoT, and error paths.

    The CoT cases embed ``<think>…</think>`` inline in the content; the
    captioner's ThinkingMixin must still strip them (predict() and
    predict_stream() share the client's streaming path, so both use the SSE
    fixtures).
    """

    @pytest.mark.asyncio
    async def test_predict(self, llamacpp, load_test_data):
        captioner: APICaptioner = llamacpp("streaming/llamacpp.txt", "llamacpp/gemma-3-27b")
        await captioner.load_model("llamacpp/gemma-3-27b")

        expected = load_test_data("streaming/llamacpp_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_streaming(self, llamacpp, load_test_data):
        captioner: APICaptioner = llamacpp("streaming/llamacpp.txt", "llamacpp/gemma-3-27b")
        await captioner.load_model("llamacpp/gemma-3-27b")

        expected = load_test_data("streaming/llamacpp_result.txt")
        got = "".join([token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))])

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_predict_with_cot(self, llamacpp, load_test_data):
        """CoT (chain-of-thought) output has inline ``<think>`` blocks that must be stripped."""
        captioner: APICaptioner = llamacpp("streaming/llamacpp_cot.txt", "llamacpp/gemma-3-27b")
        await captioner.load_model("llamacpp/gemma-3-27b")

        expected = load_test_data("streaming/llamacpp_cot_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_streaming_with_cot(self, llamacpp, load_test_data):
        """CoT output streamed token-by-token must also have ``<think>`` stripped."""
        captioner: APICaptioner = llamacpp("streaming/llamacpp_cot.txt", "llamacpp/gemma-3-27b")
        await captioner.load_model("llamacpp/gemma-3-27b")

        expected = load_test_data("streaming/llamacpp_cot_result.txt")
        got = "".join([token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))])

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_raises_error_on_bad_model(self, llamacpp):
        captioner: APICaptioner = llamacpp("streaming/llamacpp.txt", "llamacpp/gemma-3-27b")

        with pytest.raises(ValueError, match=re.compile("model not found: .*")):
            await captioner.load_model("llamacpp/unknown")
