import re

import mock
import pytest

from tests.captioners.api.conftest import make_captioner
from yadc.captioners.api import APICaptioner
from yadc.captioners.api.api_captioner import APITypes
from yadc.core import DatasetImage


@pytest.fixture
def koboldcpp(load_test_data):
    def _koboldcpp(case: str, model: str, loaded: bool = True, base_url: str = "mock://koboldcpp"):
        from tests.captioners.api.conftest import MockAsyncSession

        session = MockAsyncSession()
        loaded_model = model if loaded else "inactive"

        def _models_json(method: str, path: str, **kwargs):
            return {"data": [{"id": loaded_model, "object": "model", "owned_by": "koboldcpp"}]}

        def _api_model_json(method: str, path: str, **kwargs):
            return {"result": loaded_model}

        def _reload_json(method: str, path: str, **kwargs):
            nonlocal loaded_model
            data = kwargs.get("json", {})
            filename = data.get("filename", "")
            loaded_model = "inactive" if filename == "unload_model" else filename
            return {"success": filename != "unload_model"}

        session.register_uri("GET", "models", json=_models_json)
        session.register_uri("GET", "api/v1/model", json=_api_model_json)
        session.register_uri("GET", "api/admin/list_options", json=["unload_model", model])
        session.register_uri("POST", "api/admin/reload_config", json=_reload_json)
        session.register_uri("POST", "chat/completions", text=load_test_data(case))

        captioner, _ = make_captioner(APITypes.KOBOLDCPP, api_url=f"{base_url}/v1", session=session)
        return captioner

    return _koboldcpp


class TestKoboldcpp:
    """KoboldCpp captioner — basic prediction, streaming, model loading, and error paths.

    Note: ``predict()`` and ``predict_stream()`` share the client's streaming
    code path (the client is stream-only; ``predict`` just collects), so both
    point at the SSE fixtures.
    """

    @pytest.mark.asyncio
    async def test_predict(self, koboldcpp, load_test_data):
        captioner: APICaptioner = koboldcpp("streaming/koboldcpp.txt", "koboldcpp/gemma-3-27b")
        await captioner.load_model("koboldcpp/gemma-3-27b")

        expected = load_test_data("streaming/koboldcpp_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_streaming(self, koboldcpp, load_test_data):
        captioner: APICaptioner = koboldcpp("streaming/koboldcpp.txt", "koboldcpp/gemma-3-27b")
        await captioner.load_model("koboldcpp/gemma-3-27b")

        expected = load_test_data("streaming/koboldcpp_result.txt")
        got = "".join([token async for token in captioner.predict_stream(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))])

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_load_model_when_not_loaded(self, koboldcpp, load_test_data):
        """If the model isn't pre-loaded, ``load_model`` activates it via the admin API."""
        captioner: APICaptioner = koboldcpp("streaming/koboldcpp.txt", "koboldcpp/gemma-3-27b", loaded=False)
        await captioner.load_model("koboldcpp/gemma-3-27b")

        expected = load_test_data("streaming/koboldcpp_result.txt")
        got = await captioner.predict(mock.MagicMock(spec=DatasetImage, path="test_image.jpg"))

        assert got == expected, "bad prediction"

    @pytest.mark.asyncio
    async def test_raises_error_on_bad_model(self, koboldcpp):
        captioner: APICaptioner = koboldcpp("streaming/koboldcpp.txt", "koboldcpp/gemma-3-27b", loaded=False)

        with pytest.raises(ValueError, match=re.compile("model not found: .*")):
            await captioner.load_model("koboldcpp/unknown")
