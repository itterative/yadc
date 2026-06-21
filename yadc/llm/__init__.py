"""``yadc.llm`` — backend-agnostic LLM client layer.

Talks to OpenAI, OpenRouter, Gemini, llama.cpp, KoboldCpp, vLLM, and
Ollama through a single ``BaseLLMClient`` interface (built on the shared
:class:`AsyncSession`, :class:`HTTPResponseCache`, and
:class:`ResponseLogger` infra that also live in this package).
:class:`yadc.captioners.api.APICaptioner` composes a client to do
captioning; the prompt generator calls ``create_client`` directly.

Typical use::

    from yadc.llm import create_client, Message

    client = await create_client(api_url=..., api_token=...)
    await client.load_model(model_name)

    messages = [Message(role="system", content="..."), Message(role="user", content="...")]
    stream = await client.predict_next_message_stream(messages)
    async for chunk in stream:
        ...  # chunk.text / chunk.reasoning
    message = await stream.collect()

    await client.aclose()
"""

from .base import APIUsage, BaseLLMClient
from .factory import create_client
from .gemini import GeminiLLMClient
from .koboldcpp import KoboldcppLLMClient
from .llamacpp import LlamacppLLMClient
from .ollama import OllamaLLMClient
from .openai import OpenAILLMClient
from .openai_compatible import OpenAICompatibleLLMClient
from .openrouter import OpenRouterLLMClient
from .types import ContentPart, ImageUrl, ImageUrlPart, Message, MessageStream, StreamChunk, TextPart
from .vllm import VllmLLMClient

__all__ = [
    "APIUsage",
    "BaseLLMClient",
    "ContentPart",
    "GeminiLLMClient",
    "ImageUrl",
    "ImageUrlPart",
    "KoboldcppLLMClient",
    "LlamacppLLMClient",
    "Message",
    "MessageStream",
    "OllamaLLMClient",
    "OpenAICompatibleLLMClient",
    "OpenAILLMClient",
    "OpenRouterLLMClient",
    "StreamChunk",
    "TextPart",
    "VllmLLMClient",
    "create_client",
]
