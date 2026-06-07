"""``KoboldcppCaptioner`` — KoboldCpp backend.

``list_models(cache_ttl=...)`` queries ``/api/admin/list_options`` (the
admin endpoint that enumerates the on-disk ``.kcpps`` files) and
returns the model filenames, filtering out ``unload_model``. The result
is intentionally not cached: ``cache_ttl`` is accepted for API parity
with the other backends but the admin endpoint is not cached at the
``AsyncSession`` layer either.

``_load_model(model_repo)`` is the bespoke model-loading path
(``BaseAPICaptioner``'s default is a no-op for OpenAI-compatible
servers). It first checks ``/api/v1/model`` for the currently loaded
model (skip if it already matches ``model_repo`` or its
``koboldcpp/``-prefixed alias), then enumerates ``list_models()`` to
find a match (also accepting ``<repo>.kcpps`` filenames), and finally
``POST``s to ``/api/admin/reload_config`` with the chosen filename
and polls ``/api/v1/model`` every 500 ms until the new model is active
or the ``timeout`` (default 60 s) elapses.

``conversation()`` adds KoboldCpp's stop token (``<|im_end|>``) and
renames ``max_completion_tokens`` to ``max_tokens`` to match the local
server's request schema.
"""

import asyncio
import time
from typing import Any

import httpx
from typing_extensions import override

from yadc.core import DatasetImage, logging

from .openai import APITypes, OpenAICaptioner
from .types import (
    KoboldAdminCurrentModelResponse,
    KoboldAdminReloadModelReponse,
    KoboldAdminSettingsReponse,
)

_logger = logging.get_logger(__name__)


class KoboldcppCaptioner(OpenAICaptioner):
    _current_model: str | None

    def __init__(self, **kwargs: Any):
        super().__init__(api_type=APITypes.KOBOLDCPP, **kwargs)
        self._current_model = None

    @override
    async def _load_model(self, model_repo: str, timeout: float = 60):
        assert self._async_session is not None, "async session not available"

        if self._current_model == model_repo:
            return

        model_koboldcpp = "koboldcpp/" + model_repo
        model_kcpss = model_repo + ".kcpps"

        # early exit if already loaded
        async with self._async_session.get("/api/v1/model") as model_current_resp:
            assert isinstance(model_current_resp, httpx.Response)
            assert model_current_resp.status_code < 400

            model_current_resp_json = model_current_resp.json()
            assert isinstance(model_current_resp_json, dict)

            model_current = KoboldAdminCurrentModelResponse.model_validate(model_current_resp_json)

            if model_current.result == model_repo or model_current.result == model_koboldcpp:
                self._current_model = model_current.result
                return

        available_models = await self.list_models()

        for model in available_models:
            if model == model_repo or model == model_kcpss:
                self._current_model = model
                break

        if not self._current_model:
            if available_models:
                raise ValueError(f"model not found: {model_repo}; available models: {', '.join(available_models)}")

            raise ValueError(f"model not found: {model_repo}; no models available")

        async with self._async_session.post("/api/admin/reload_config", json={"filename": self._current_model}) as model_reload_resp:
            assert isinstance(model_reload_resp, httpx.Response)
            assert model_reload_resp.status_code < 400

            model_reload_resp_json = model_reload_resp.json()
            assert isinstance(model_reload_resp_json, dict)

            if not KoboldAdminReloadModelReponse.model_validate(model_reload_resp_json).success:
                raise ValueError(f"failed to load model: {model_repo}")

        start_t = time.time()
        end_t = start_t + timeout

        while time.time() < end_t:
            try:
                async with self._async_session.get("/api/v1/model") as model_current_resp:
                    assert isinstance(model_current_resp, httpx.Response)
                    assert model_current_resp.status_code < 400

                    model_current_resp_json = model_current_resp.json()
                    assert isinstance(model_current_resp_json, dict)

                    model_current = KoboldAdminCurrentModelResponse.model_validate(model_current_resp_json)

                    if model_current.result == "inactive":
                        await asyncio.sleep(0.5)
                        continue

                    self._current_model = model_current.result
                    break
            except (httpx.ConnectError, httpx.TimeoutException):
                await asyncio.sleep(0.5)
                continue
        else:
            raise TimeoutError(f"failed to load model in time: {model_repo}")

    @override
    async def list_models(self, cache_ttl: float | None = None) -> list[str]:
        """Fetch the list of available model filenames from Koboldcpp's admin endpoint.

        Koboldcpp's model list is not cached — the admin endpoint returns
        the current on-disk state and the *cache_ttl* parameter is ignored.
        """
        if cache_ttl is not None:
            _logger.debug("Koboldcpp list_models ignores cache_ttl=%s (admin endpoint is not cached).", cache_ttl)

        assert self._async_session is not None, "async session not available"

        async with self._async_session.get("/api/admin/list_options") as model_options_resp:
            assert isinstance(model_options_resp, httpx.Response)
            assert model_options_resp.status_code < 400

            model_options_resp_json = model_options_resp.json()
            assert isinstance(model_options_resp_json, list), "bad koboldcpp list_options response"

            models = KoboldAdminSettingsReponse.model_validate({"data": model_options_resp_json})

            return [m for m in models.data if m != "unload_model"]

    @override
    def conversation(self, image: DatasetImage, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        conversation: dict[str, Any] = super().conversation(image, stream=stream, **kwargs)

        # reference: https://github.com/LostRuins/koboldcpp/blob/575eb4095095939b016dc2e1957643ffb2dbf086/tools/server/bench/script.js#L98
        conversation.setdefault("stop", ["<|im_end|>"])

        conversation["max_tokens"] = conversation.pop("max_completion_tokens", 512)

        return conversation
