from typing import Literal, Optional

import pydantic

class OpenAIModelsResponse(pydantic.BaseModel):
    object: Literal['list']
    data: list['OpenAIModel']

class OpenAIModel(pydantic.BaseModel):
    id: str
    object: Literal['model']
    owned_by: str = ''

class OpenAIStreamingResponse(pydantic.BaseModel):
    error: Optional['_OpenAIStreamingError'] = None
    choices: list['_OpenAIStreamingChoice'] = []

class _OpenAIStreamingError(pydantic.BaseModel):
    code: str
    message: str

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
    candidates: list['_GeminiStreamingCandidate']

class _GeminiStreamingCandidate(pydantic.BaseModel):
    content: '_GeminiStreamingCandidateContent'

class _GeminiStreamingCandidateContent(pydantic.BaseModel):
    parts: list['_GeminiStreamingCandidatePart']

class _GeminiStreamingCandidatePart(pydantic.BaseModel):
    text: str = ''

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
