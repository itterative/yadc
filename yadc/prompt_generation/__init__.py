"""``yadc.prompt_generation`` — generate Jinja2 prompt templates via an LLM.

Neutral, layer-agnostic package that owns the prompt-generation
domain: request models, the meta-conversation scaffolding, and the
streaming core that drives an LLM client. Both the CLI
(``yadc.cmd.prompts``) and the API (``yadc.api.services.prompt_generation.PromptGenerationService``)
call into the streaming core here, so the env-load → validate →
client-build → stream → cleanup wiring lives in exactly one place.

Layering: this package is the source of truth and may be extended
with additional entry points in the future. It imports from
``yadc.cmd.envs`` (env loading) and ``yadc.llm`` (client layer); the
API service is a thin DI wrapper that delegates to it.
"""

from .models import (
    ExamplePair,
    PromptGenerationConfigError,
    PromptGenerationError,
    PromptGenerationFocus,
    PromptGenerationRequest,
)
from .streaming import stream_template_chunks

__all__ = [
    "ExamplePair",
    "PromptGenerationConfigError",
    "PromptGenerationError",
    "PromptGenerationFocus",
    "PromptGenerationRequest",
    "stream_template_chunks",
]
