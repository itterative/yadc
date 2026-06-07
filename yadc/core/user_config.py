"""``UserConfig`` / ``UserConfigApi`` Pydantic models (legacy — replaced by ``AppConfig`` for envs and ``Config`` for dataset configs)."""

import pydantic


class UserConfig(pydantic.BaseModel):
    api: "UserConfigApi"


class UserConfigApi(pydantic.BaseModel):
    url: str = ""
    token: str = ""
    model_name: str = ""
    max_concurrent: int | None = None
