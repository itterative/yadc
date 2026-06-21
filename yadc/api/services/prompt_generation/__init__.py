"""``PromptGenerationService`` — package entry point.

Re-exports the public surface and (for test-patch compatibility)
``cmd_envs`` + ``create_client`` at the package level. The actual
service + message-building logic lives in :mod:`.service`.
"""

from yadc.cmd import envs as cmd_envs
from yadc.llm import create_client

from .service import (
    ExamplePair,
    PromptGenerationFocus,
    PromptGenerationRequest,
    PromptGenerationService,
)

__all__ = [
    "ExamplePair",
    "PromptGenerationFocus",
    "PromptGenerationRequest",
    "PromptGenerationService",
    "cmd_envs",
    "create_client",
]
