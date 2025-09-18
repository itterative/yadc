from typing import Literal, Optional

import pydantic

class OpenAIModelsResponse(pydantic.BaseModel):
    object: Literal['list']
    data: list['OpenAIModel']

class OpenAIModel(pydantic.BaseModel):
    id: str
    object: Literal['model']
    owned_by: str = ''

class GeminiModelsResponse(pydantic.BaseModel):
    models: list['GeminiModel']
    nextPageToken: Optional[str] = None

class GeminiModel(pydantic.BaseModel):
    name: str
    version: str
    displayName: str
    supportedGenerationMethods: list[str]
    thinking: bool = False

class KoboldAdminCurrentModelResponse(pydantic.BaseModel):
    result: str

class KoboldAdminSettingsReponse(pydantic.BaseModel):
    data: list[str]

class KoboldAdminReloadModelReponse(pydantic.BaseModel):
    success: bool
