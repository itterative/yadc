import pydantic

class UserConfig(pydantic.BaseModel):
    api: 'UserConfigApi'

class UserConfigApi(pydantic.BaseModel):
    url: str = ''
    token: str = ''
    model_name: str = ''
