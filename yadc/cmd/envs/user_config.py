"""Legacy ``UserConfig`` / ``UserConfigApi`` Pydantic models (replaced by ``AppConfig`` for envs)."""

import pydantic


class UserConfig(pydantic.BaseModel):
    api: "UserConfigApi"


class UserConfigApi(pydantic.BaseModel):
    url: str = ""
    token: str = ""
    model_name: str = ""
    max_concurrent: int | None = None
