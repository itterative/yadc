from typing import Literal, Optional

import pydantic

class OpenAIModelsResponse(pydantic.BaseModel):
    data: list['OpenAIModel']

class OpenAIModel(pydantic.BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: str = 'default'

class OpenAIStreamingResponse(pydantic.BaseModel):
    id: str = 'SKIPPED'
    error: Optional['_OpenAIStreamingError'] = None
    choices: list['_OpenAIStreamingChoice'] = []
    usage: Optional['_OpenAIUsage'] = None

class _OpenAIStreamingError(pydantic.BaseModel):
    code: str
    message: str
    metadata: Optional[dict] = None

class _OpenAIStreamingChoice(pydantic.BaseModel):
    delta: '_OpenAIStreamingChoiceDelta'
    finish_reason: Optional[str] = None

class _OpenAIStreamingChoiceDelta(pydantic.BaseModel):
    content: str = ''

class OpenAIErrorResponse(pydantic.BaseModel):
    error: '_OpenAIError'

class _OpenAIError(pydantic.BaseModel):
    code: int
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

class GeminiStreamingResponse(pydantic.BaseModel):
    responseId: str = 'SKIPPED'
    candidates: list['_GeminiStreamingCandidate']
    promptFeedback: Optional['_GeminiStreamingPromptFeedback'] = None
    usageMetadata: Optional['_GeminiStreamingUsageMetadata'] = None

class _GeminiStreamingCandidate(pydantic.BaseModel):
    content: '_GeminiStreamingCandidateContent'
    finishReason: Optional[str] = None

class _GeminiStreamingCandidateContent(pydantic.BaseModel):
    parts: list['_GeminiStreamingCandidatePart']

class _GeminiStreamingCandidatePart(pydantic.BaseModel):
    text: str = ''
    thought: bool = False

class _GeminiStreamingPromptFeedback(pydantic.BaseModel):
    blockReason: str = 'BLOCK_REASON_UNSPECIFIED'
    safetyRatings: list[str] = []

class _GeminiStreamingUsageMetadata(pydantic.BaseModel):
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
