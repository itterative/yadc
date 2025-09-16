from typing import Literal

import pydantic

class OpenAIModelsResponse(pydantic.BaseModel):
    object: Literal['list']
    data: list['OpenAIModel']

class OpenAIModel(pydantic.BaseModel):
    id: str
    object: Literal['model']
    owned_by: str = ''


class KoboldAdminCurrentModelResponse(pydantic.BaseModel):
    result: str

class KoboldAdminSettingsReponse(pydantic.BaseModel):
    data: list[str]

class KoboldAdminReloadModelReponse(pydantic.BaseModel):
    success: bool
