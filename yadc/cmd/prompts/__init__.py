"""Pure logic for ``yadc prompts`` — prompt template generation via LLM."""

from .prompts import generate, resolve_examples_targets

__all__ = ["generate", "resolve_examples_targets"]
