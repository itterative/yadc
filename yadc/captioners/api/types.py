from typing import Literal, Optional

import pydantic

class OpenAIModelsResponse(pydantic.BaseModel):
    data: list['OpenAIModel']

class OpenAIModel(pydantic.BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: str = 'default'

class OpenAIChatCompletionChunkResponse(pydantic.BaseModel):
    id: str = 'SKIPPED'
    object: str = 'chat.completion.cunk'
    error: Optional['_OpenAIChatCompletionError'] = None
    choices: list['_OpenAIChatCompletionChunkChoice'] = []
    usage: Optional['_OpenAIUsage'] = None

class OpenAIChatCompletionResponse(pydantic.BaseModel):
    id: str = 'SKIPPED'
    object: str = 'chat.completion'
    choices: list['_OpenAIChatCompletionChoice'] = []
    usage: Optional['_OpenAIUsage'] = None

class _OpenAIChatCompletionError(pydantic.BaseModel):
    code: str
    message: str
    metadata: Optional[dict] = None

class _OpenAIChatCompletionChunkChoice(pydantic.BaseModel):
    delta: '_OpenAIChatCompletionChunkChoiceDelta'
    finish_reason: Optional[str] = None

class _OpenAIChatCompletionChoice(pydantic.BaseModel):
    message: '_OpenAIChatCompletionChoiceMessage'
    finish_reason: Optional[str] = None

class _OpenAIChatCompletionChoiceMessage(pydantic.BaseModel):
    role: str = 'assistant'
    refusal: Optional[str] = None
    content: Optional[str] = None

class _OpenAIChatCompletionChunkChoiceDelta(pydantic.BaseModel):
    role: str = 'assistant'
    refusal: Optional[str] = None
    content: Optional[str] = None

class OpenAIErrorResponse(pydantic.BaseModel):
    error: '_OpenAIError'

class _OpenAIError(pydantic.BaseModel):
    code: int|str
    message: str
    metadata: Optional[dict] = None

class _OpenAIUsage(pydantic.BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Optional['_OpenAIUsageCompletionDetails'] = None

class _OpenAIUsageCompletionDetails(pydantic.BaseModel):
    reasoning_tokens: int = 0

class GeminiModelsResponse(pydantic.BaseModel):
    models: list['GeminiModel']
    nextPageToken: Optional[str] = None

class GeminiModel(pydantic.BaseModel):
    name: str
    version: str
    displayName: str
    supportedGenerationMethods: list[str]
    thinking: bool = False

class GeminiErrorResponse(pydantic.BaseModel):
    error: '_GeminiError'

class _GeminiError(pydantic.BaseModel):
    code: int
    message: str
    status: str

class GeminiContentResponse(pydantic.BaseModel):
    responseId: str = 'SKIPPED'
    candidates: list['_GeminiContentCandidate']
    promptFeedback: Optional['_GeminiContentPromptFeedback'] = None
    usageMetadata: Optional['_GeminiContentUsageMetadata'] = None

class _GeminiContentCandidate(pydantic.BaseModel):
    content: '_GeminiContentCandidateContent'
    finishReason: Optional[str] = None

class _GeminiContentCandidateContent(pydantic.BaseModel):
    parts: list['_GeminiContentCandidatePart']

class _GeminiContentCandidatePart(pydantic.BaseModel):
    text: str = ''
    thought: bool = False

class _GeminiContentPromptFeedback(pydantic.BaseModel):
    blockReason: str = 'BLOCK_REASON_UNSPECIFIED'
    safetyRatings: list[str] = []

class _GeminiContentUsageMetadata(pydantic.BaseModel):
    candidatesTokenCount: int = 0
    promptTokenCount: int = 0
    totalTokenCount: int = 0
    thoughtsTokenCount: int = 0

class OpenRouterCreditsResponse(pydantic.BaseModel):
    data: '_OpenRouterCredits'

class _OpenRouterCredits(pydantic.BaseModel):
    total_credits: float
    total_usage: float

class OpenRouterModerationError(pydantic.BaseModel):
    reasons: list[str] = []
    flagged_input: str = ''
    provider_name: str = ''
    model_slug: str = ''

class KoboldServiceInfoResponse(pydantic.BaseModel):
    software: '_KoboldServiceInfoSoftware'

class _KoboldServiceInfoSoftware(pydantic.BaseModel):
    name: str

class KoboldAdminCurrentModelResponse(pydantic.BaseModel):
    result: str

class KoboldAdminSettingsReponse(pydantic.BaseModel):
    data: list[str]

class KoboldAdminReloadModelReponse(pydantic.BaseModel):
    success: bool
