"""``OpenAILLMClient`` — thin client for OpenAI's hosted API.

Identical wire format to :class:`OpenAICompatibleLLMClient`; exists to give
the right type identity for :func:`yadc.llm.create_client` (and as the
obvious import for OpenAI callers). Cloud ``load_model`` is the base
existence check.
"""

from .openai_compatible import OpenAICompatibleLLMClient


class OpenAILLMClient(OpenAICompatibleLLMClient):
    """Client for the OpenAI hosted API (``api.openai.com``)."""
