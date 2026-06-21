"""``OpenRouterLLMClient`` — OpenRouter backend.

OpenRouter uses its own ``reasoning`` config shape (``{enabled, effort,
exclude}``) instead of OpenAI's ``reasoning_effort``, swaps
``max_completion_tokens`` for ``max_tokens``, prefers a top-level
``usage: {include: true}`` over ``stream_options``, and redacts reasoning
text to ``"[REDACTED]"`` — which we must detect so it isn't surfaced as
real reasoning.
"""

from typing import Any, Literal

import httpx
from typing_extensions import override

from yadc.core import logging

from .openai_compatible import OpenAICompatibleLLMClient
from .response_models import OpenRouterCreditsResponse

_logger = logging.get_logger(__name__)


class OpenRouterLLMClient(OpenAICompatibleLLMClient):
    """Client for the OpenRouter hosted API (``openrouter.ai``)."""

    @override
    async def load_model(self, model_repo: str, **kwargs: Any) -> None:
        await super().load_model(model_repo, **kwargs)
        await self._alog_api_information()

    async def _alog_api_information(self) -> None:
        """Log remaining credits after loading the model.

        OpenRouter exposes account credit usage at ``GET /credits``; surface
        it so callers know how much budget remains. Failures are non-fatal —
        the model is already loaded — so we just warn.
        """
        try:
            async with self._async_session.get("credits") as credits_resp:
                assert isinstance(credits_resp, httpx.Response)
                credits_resp.raise_for_status()

                credits_resp_json = credits_resp.json()
                assert isinstance(credits_resp_json, dict)

                credits = OpenRouterCreditsResponse.model_validate(credits_resp_json).data
                _logger.info("You have used %.2f out of %.2f credits with this api token.", credits.total_usage, credits.total_credits)
        except Exception:
            _logger.warning("Warning: failed to retrieve current credits. Is your API token correct?")

    @override
    @staticmethod
    def _is_reasoning_redacted(text: str) -> bool:
        return text == "[REDACTED]"

    @override
    def _customize_body(self, body: dict[str, Any]) -> dict[str, Any]:
        body = super()._customize_body(body)

        # OpenRouter uses its own reasoning block instead of reasoning_effort.
        reasoning_effort = body.pop("reasoning_effort", None)
        reasoning_exclude = body.pop("_reasoning_exclude", False)
        if reasoning_effort is None:
            body["reasoning"] = {"enabled": False}
        else:
            _effort: Literal["low", "medium", "high"] = reasoning_effort  # type: ignore[assignment]
            body["reasoning"] = {"enabled": True, "effort": _effort, "exclude": reasoning_exclude}

        body.pop("stream_options", None)
        body["usage"] = {"include": True}

        body["max_tokens"] = body.pop("max_completion_tokens", None)

        return body
