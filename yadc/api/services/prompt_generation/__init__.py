"""``PromptGenerationService`` — package entry point.

Re-exports the public surface from :mod:`yadc.prompt_generation` for
backward compatibility. The actual implementation lives in the
neutral ``yadc.prompt_generation`` package; this module exists so the
existing import path
``from yadc.api.services.prompt_generation import PromptGenerationRequest``
continues to work.
"""

from yadc.prompt_generation import (
    ExamplePair,
    PromptGenerationConfigError,
    PromptGenerationError,
    PromptGenerationFocus,
    PromptGenerationRequest,
)

from .service import PromptGenerationService

__all__ = [
    "ExamplePair",
    "PromptGenerationConfigError",
    "PromptGenerationError",
    "PromptGenerationFocus",
    "PromptGenerationRequest",
    "PromptGenerationService",
]
