from typing import Any

import httpx
from typing_extensions import override

from yadc.core import DatasetImage, logging

from .openai import APITypes, OpenAICaptioner
from .types import OpenRouterCreditsResponse

_logger = logging.get_logger(__name__)


class OpenRouterCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs: Any):
        super().__init__(api_type=APITypes.OPENROUTER, **kwargs)

    @override
    async def load_model(self, model_repo: str, **kwargs: Any) -> None:
        await super().load_model(model_repo, **kwargs)
        await self._alog_api_information()

    async def _alog_api_information(self):
        assert self._async_session is not None, "async session not available"

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
    def conversation(self, image: DatasetImage, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        conversation: dict[str, Any] = super().conversation(image, stream=stream, **kwargs)

        conversation.pop("reasoning_effort", None)  # remove any existing openai reasoning config

        if self._reasoning:
            conversation["reasoning"] = {
                "enabled": True,
                "effort": self._reasoning_effort,
                "exclude": self._reasoning_exclude_output,
            }
        else:
            conversation["reasoning"] = {"enabled": False}

        conversation.pop("stream_options", None)
        conversation["usage"] = {"include": True}

        conversation["max_tokens"] = conversation.pop("max_completion_tokens", 512)

        return conversation
