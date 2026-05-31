import re

import mock
import pytest

from tests.captioners.api.conftest import MockAsyncSession
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


@pytest.fixture
def llamacpp(load_test_data):
    def _llamacpp(case: str, model: str, base_url: str = "mock://llamacpp"):
        session = MockAsyncSession()
        session.register_uri(
            "GET",
            "models",
            json={"data": [{"id": model, "object": "model", "owned_by": "llamacpp"}]},
        )
        session.register_uri("POST", "chat/completions", text=load_test_data(case))

        captioner = APICaptioner(
            api_type=APITypes.LLAMACPP,
            api_url=f"{base_url}/v1",
            async_session=session,
        )

        return captioner

    return _llamacpp


@pytest.mark.asyncio
async def test_llamacpp(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp("nonstreaming/llamacpp.txt", "llamacpp/gemma-3-27b")
    await captioner.load_model("llamacpp/gemma-3-27b")

    expected = load_test_data("nonstreaming/llamacpp_result.txt")
    got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

    assert got == expected, "bad prediction"


@pytest.mark.asyncio
async def test_llamacpp_streaming(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp("streaming/llamacpp.txt", "llamacpp/gemma-3-27b")
    await captioner.load_model("llamacpp/gemma-3-27b")

    expected = load_test_data("streaming/llamacpp_result.txt")
    got = "".join([ token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg")) ])

    assert got == expected, "bad prediction"


@pytest.mark.asyncio
async def test_llamacpp_cot(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp("nonstreaming/llamacpp_cot.txt", "llamacpp/gemma-3-27b")
    await captioner.load_model("llamacpp/gemma-3-27b")

    expected = load_test_data("nonstreaming/llamacpp_cot_result.txt")
    got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

    assert got == expected, "bad prediction"


@pytest.mark.asyncio
async def test_llamacpp_streaming_cot(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp("streaming/llamacpp_cot.txt", "llamacpp/gemma-3-27b")
    await captioner.load_model("llamacpp/gemma-3-27b")

    expected = load_test_data("streaming/llamacpp_cot_result.txt")
    got = "".join([ token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg")) ])

    assert got == expected, "bad prediction"


@pytest.mark.asyncio
async def test_llamacpp_should_raise_error_on_bad_model(llamacpp):
    captioner: APICaptioner = llamacpp("nonstreaming/llamacpp.txt", "llamacpp/gemma-3-27b")

    with pytest.raises(ValueError, match=re.compile("model not found: .*")):
        await captioner.load_model("llamacpp/unknown")
