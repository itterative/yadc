"""``LlamacppLLMClient`` — llama.cpp backend.

llama.cpp's server expects ``max_tokens`` (not ``max_completion_tokens``).
Everything else is plain OpenAI-compatible.
"""

from typing import Any

from typing_extensions import override

from .openai_compatible import OpenAICompatibleLLMClient


class LlamacppLLMClient(OpenAICompatibleLLMClient):
    """Client for a llama.cpp server's OpenAI-compatible endpoint."""

    @override
    def _customize_body(self, body: dict[str, Any]) -> dict[str, Any]:
        body = super()._customize_body(body)
        body["max_tokens"] = body.pop("max_completion_tokens", None)
        return body
