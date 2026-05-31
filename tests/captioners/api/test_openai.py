import re

import mock
import pytest

from tests.captioners.api.conftest import MockAsyncSession
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


@pytest.fixture
def openai(load_test_data):
    def _openai(case: str, model: str, base_url: str = "mock://api.openai.com/v1"):
        session = MockAsyncSession()
        session.register_uri(
            "GET",
            "models",
            json={"data": [{"id": model, "object": "model", "owned_by": "openai"}]},
        )
        session.register_uri("POST", "chat/completions", text=load_test_data(case))

        captioner = APICaptioner(
            api_type=APITypes.OPENAI,
            api_url=base_url,
            api_token="api_token",
            async_session=session,
        )

        return captioner

    return _openai


@pytest.mark.asyncio
async def test_openai_o4_mini(openai, load_test_data):
    captioner: APICaptioner = openai("nonstreaming/openai_o4_mini.txt", "o4-mini")
    await captioner.load_model("o4-mini")

    expected = load_test_data("nonstreaming/openai_o4_mini_result.txt")
    got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

    assert got == expected, "bad prediction"


@pytest.mark.asyncio
async def test_openai_o4_mini_streaming(openai, load_test_data):
    captioner: APICaptioner = openai("streaming/openai_o4_mini.txt", "o4-mini")
    await captioner.load_model("o4-mini")

    expected = load_test_data("streaming/openai_o4_mini_result.txt")
    got = "".join([ token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg")) ])

    assert got == expected, "bad prediction"


@pytest.mark.asyncio
async def test_openai_raises_error_on_bad_model(openai):
    captioner: APICaptioner = openai("nonstreaming/openai_o4_mini.txt", "o4-mini")

    with pytest.raises(ValueError, match=re.compile("model not found: .*")):
        await captioner.load_model("unknown")
