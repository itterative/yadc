"""Re-exports the captioning-side ``APICaptioner``.

``APICaptioner`` composes a :class:`yadc.llm.BaseLLMClient` and holds all
captioning-specific glue (image encoding, Jinja prompts, conversation
overrides, thinking strip). Build the client with
:func:`yadc.llm.create_client`, then wrap it::

    from yadc.llm import create_client
    from yadc.captioners.api import APICaptioner

    client = await create_client(api_url=..., api_token=...)
    captioner = APICaptioner(client=client, prompt_template=..., ...)
"""

from .api_captioner import APICaptioner, APITypes

__all__ = [
    "APICaptioner",
    "APITypes",
]
