"""Pydantic models for OpenAI, Gemini, OpenRouter, KoboldCpp, and llama.cpp API response types.

The captioners use these to parse upstream responses in a uniform way
rather than hand-rolling dict access. Most models are ``extra="allow"``
so unknown fields don't fail parsing, and a handful expose ``usage``
and ``usageMetadata`` as optional sub-models so token accounting is a
no-op when the backend omits it.

Grouped by backend:

- **OpenAI** — ``OpenAIChatCompletionResponse`` (choices + usage),
  ``OpenAIChatCompletionChunkResponse`` (streaming delta), the
  ``OpenAIModelsResponse`` / ``OpenAIModel`` list-models pair, the
  error shape (``OpenAIErrorResponse``), and the ``reasoning.details``
  variants (``text`` / ``summary`` / ``encrypted``).
- **Gemini** — ``GeminiContentResponse`` with ``candidates``,
  ``parts`` (text + ``thought: true`` reasoning), and
  ``usageMetadata``; ``GeminiModelsResponse`` / ``GeminiModel`` for
  the paginated ``models.list``; ``GeminiErrorResponse``.
- **OpenRouter** — ``OpenRouterModerationError`` and
  ``OpenRouterCreditsResponse`` (used by ``OpenRouterCaptioner`` to log
  remaining credits after a successful ``load_model``).
- **KoboldCpp** — admin endpoints: ``KoboldAdminCurrentModelResponse``,
  ``KoboldAdminReloadModelReponse``, ``KoboldAdminSettingsReponse``, and
  ``KoboldServiceInfoResponse`` (used by ``APICaptioner._infer_api_type``).
"""

from typing import Any, ClassVar, Literal

import pydantic


class OpenAIModelsResponse(pydantic.BaseModel):
    data: list["OpenAIModel"]


class OpenAIModel(pydantic.BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "default"


class OpenAIChatCompletionChunkResponse(pydantic.BaseModel):
    id: str = "SKIPPED"
    object: str = "chat.completion.cunk"
    error: "_OpenAIChatCompletionError | None" = None
    choices: list["_OpenAIChatCompletionChunkChoice"] = []
    usage: "_OpenAIUsage | None" = None


class OpenAIChatCompletionResponse(pydantic.BaseModel):
    id: str = "SKIPPED"
    object: str = "chat.completion"
    choices: list["_OpenAIChatCompletionChoice"] = []
    usage: "_OpenAIUsage | None" = None


class _OpenAIChatCompletionError(pydantic.BaseModel):
    code: str
    message: str
    metadata: dict[str, Any] | None = None


class _OpenAIChatCompletionChunkChoice(pydantic.BaseModel):
    delta: "_OpenAIChatCompletionChunkChoiceDelta"
    finish_reason: str | None = None


class _OpenAIChatCompletionChoice(pydantic.BaseModel):
    message: "_OpenAIChatCompletionChoiceMessage"
    finish_reason: str | None = None


class _OpenAIReasoningDetailText(pydantic.BaseModel):
    type: Literal["reasoning.text"]
    text: str = ""
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")


class _OpenAIReasoningDetailSummary(pydantic.BaseModel):
    type: Literal["reasoning.summary"]
    summary: str = ""
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")


class _OpenAIReasoningDetailEncrypted(pydantic.BaseModel):
    type: Literal["reasoning.encrypted"]
    data: str = ""
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")


_OpenAIReasoningDetail = _OpenAIReasoningDetailText | _OpenAIReasoningDetailSummary | _OpenAIReasoningDetailEncrypted


class _OpenAIChatCompletionChoiceMessage(pydantic.BaseModel):
    role: str = "assistant"
    refusal: str | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[_OpenAIReasoningDetail] | None = None


class _OpenAIChatCompletionChunkChoiceDelta(pydantic.BaseModel):
    role: str = "assistant"
    refusal: str | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[_OpenAIReasoningDetail] | None = None


class OpenAIErrorResponse(pydantic.BaseModel):
    error: "_OpenAIError"


class _OpenAIError(pydantic.BaseModel):
    code: int | str
    message: str
    metadata: dict[str, Any] | None = None


class _OpenAIUsage(pydantic.BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: "_OpenAIUsageCompletionDetails | None" = None


class _OpenAIUsageCompletionDetails(pydantic.BaseModel):
    reasoning_tokens: int = 0


class GeminiModelsResponse(pydantic.BaseModel):
    models: list["GeminiModel"]
    nextPageToken: str | None = None


class GeminiModel(pydantic.BaseModel):
    name: str
    version: str
    displayName: str
    supportedGenerationMethods: list[str]
    thinking: bool = False


class GeminiErrorResponse(pydantic.BaseModel):
    error: "_GeminiError"


class _GeminiError(pydantic.BaseModel):
    code: int
    message: str
    status: str


class GeminiContentResponse(pydantic.BaseModel):
    responseId: str = "SKIPPED"
    candidates: list["_GeminiContentCandidate"]
    promptFeedback: "_GeminiContentPromptFeedback | None" = None
    usageMetadata: "_GeminiContentUsageMetadata | None" = None


class _GeminiContentCandidate(pydantic.BaseModel):
    content: "_GeminiContentCandidateContent"
    finishReason: str | None = None


class _GeminiContentCandidateContent(pydantic.BaseModel):
    parts: list["_GeminiContentCandidatePart"]


class _GeminiContentCandidatePart(pydantic.BaseModel):
    text: str = ""
    thought: bool = False


class _GeminiContentPromptFeedback(pydantic.BaseModel):
    blockReason: str = "BLOCK_REASON_UNSPECIFIED"
    safetyRatings: list[str] = []


class _GeminiContentUsageMetadata(pydantic.BaseModel):
    candidatesTokenCount: int = 0
    promptTokenCount: int = 0
    totalTokenCount: int = 0
    thoughtsTokenCount: int = 0


class OpenRouterCreditsResponse(pydantic.BaseModel):
    data: "_OpenRouterCredits"


class _OpenRouterCredits(pydantic.BaseModel):
    total_credits: float
    total_usage: float


class OpenRouterModerationError(pydantic.BaseModel):
    reasons: list[str] = []
    flagged_input: str = ""
    provider_name: str = ""
    model_slug: str = ""


class KoboldServiceInfoResponse(pydantic.BaseModel):
    software: "_KoboldServiceInfoSoftware"


class _KoboldServiceInfoSoftware(pydantic.BaseModel):
    name: str


class KoboldAdminCurrentModelResponse(pydantic.BaseModel):
    result: str


class KoboldAdminSettingsReponse(pydantic.BaseModel):
    data: list[str]


class KoboldAdminReloadModelReponse(pydantic.BaseModel):
    success: bool
