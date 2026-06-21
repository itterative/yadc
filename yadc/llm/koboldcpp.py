"""``KoboldcppLLMClient`` — KoboldCpp backend.

KoboldCpp's server expects ``max_tokens`` (not ``max_completion_tokens``)
and a stop token (``<|im_end|>``). Its model list lives behind the admin
endpoint (``/api/admin/list_options``), and loading a model means
``POST /api/admin/reload_config`` then polling ``/api/v1/model`` until the
new model is active.
"""

import asyncio
import time
from typing import Any

import httpx
from typing_extensions import override

from yadc.core import logging

from .error_normalization import normalize_error
from .openai_compatible import OpenAICompatibleLLMClient
from .response_models import (
    KoboldAdminCurrentModelResponse,
    KoboldAdminReloadModelReponse,
    KoboldAdminSettingsReponse,
)

_logger = logging.get_logger(__name__)


class KoboldcppLLMClient(OpenAICompatibleLLMClient):
    """Client for a KoboldCpp server's OpenAI-compatible endpoint."""

    @override
    async def load_model(self, model_repo: str, timeout: float = 60, **_kwargs: Any) -> None:
        """Load (or reload) ``model_repo`` into VRAM and wait until it's active.

        No-op if it's already the current model; otherwise reloads via
        ``/api/admin/reload_config`` and polls ``/api/v1/model`` until the
        server reports the new model (or ``timeout`` seconds elapse).
        """
        try:
            self._model = await self._load_model(model_repo, timeout=timeout)
        except httpx.HTTPStatusError as e:
            raise ValueError(await normalize_error(e)) from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ValueError(f"api unavailable: {self._api_url}") from e

        _logger.info("Model set to %s.", self._model)

    async def _load_model(self, model_repo: str, *, timeout: float = 60) -> str:
        if self._model == model_repo:
            return model_repo

        model_koboldcpp = "koboldcpp/" + model_repo
        model_kcpss = model_repo + ".kcpps"

        # Fast path: the right model is already loaded.
        async with self._async_session.get("/api/v1/model") as model_current_resp:
            assert isinstance(model_current_resp, httpx.Response)
            assert model_current_resp.status_code < 400

            model_current_resp_json = model_current_resp.json()
            assert isinstance(model_current_resp_json, dict)

            model_current = KoboldAdminCurrentModelResponse.model_validate(model_current_resp_json)

            if model_current.result == model_repo or model_current.result == model_koboldcpp:
                self._model: str | None = model_current.result
                return model_current.result

        available_models = await self.list_models()

        chosen: str | None = None
        for model in available_models:
            if model == model_repo or model == model_kcpss:
                chosen = model
                break

        if chosen is None:
            if available_models:
                raise ValueError(f"model not found: {model_repo}; available models: {', '.join(available_models)}")

            raise ValueError(f"model not found: {model_repo}; no models available")

        async with self._async_session.post("/api/admin/reload_config", json={"filename": chosen}) as model_reload_resp:
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

                    return model_current.result
            except (httpx.ConnectError, httpx.TimeoutException):
                await asyncio.sleep(0.5)
                continue

        raise TimeoutError(f"failed to load model in time: {model_repo}")

    @override
    async def list_models(self, *, cache_ttl: float | None = None) -> list[str]:
        """Fetch available model filenames from KoboldCpp's admin endpoint.

        KoboldCpp's model list is not cached — the admin endpoint returns the
        current on-disk state, so *cache_ttl* is accepted for API parity but
        ignored.
        """
        if cache_ttl is not None:
            _logger.debug("Koboldcpp list_models ignores cache_ttl=%s (admin endpoint is not cached).", cache_ttl)

        async with self._async_session.get("/api/admin/list_options") as model_options_resp:
            assert isinstance(model_options_resp, httpx.Response)
            assert model_options_resp.status_code < 400

            model_options_resp_json = model_options_resp.json()
            assert isinstance(model_options_resp_json, list), "bad koboldcpp list_options response"

            models = KoboldAdminSettingsReponse.model_validate({"data": model_options_resp_json})

            return [m for m in models.data if m != "unload_model"]

    @override
    def _customize_body(self, body: dict[str, Any]) -> dict[str, Any]:
        body = super()._customize_body(body)

        # https://github.com/LostRuins/koboldcpp/blob/575eb4095095939b016dc2e1957643ffb2dbf086/tools/server/bench/script.js#L98
        body.setdefault("stop", ["<|im_end|>"])
        body["max_tokens"] = body.pop("max_completion_tokens", None)

        return body
