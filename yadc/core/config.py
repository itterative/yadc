from typing import Optional

import pydantic

from .dataset import DatasetImage

class Config(pydantic.BaseModel):
    api: 'ConfigApi' = pydantic.Field(default_factory=lambda: ConfigApi())
    settings: 'ConfigSettings' = pydantic.Field(default_factory=lambda: ConfigSettings())
    dataset: 'ConfigDataset' = pydantic.Field(default_factory=lambda: ConfigDataset())

    interactive: bool = False
    rounds: int = 1
    caption_suffix: str = '.txt'
    overwrite_captions: bool = False

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.caption_suffix.startswith('.'), f"invalid caption_suffix: {self.caption_suffix}"

            assert self.rounds > 0, 'rounds must be a positive number'
        except AssertionError as e:
            raise ValueError(e)
        
        return self

class ConfigSettings(pydantic.BaseModel):
    hf_token: str = ''
    max_tokens: int = 512

    prompt_template: str = ''
    prompt_template_path: str = 'default.jinja'

    store_conversation: bool = False
    image_quality: str = 'auto'
    debug_prompt: bool = False

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert 100 < self.max_tokens < 1200, 'config max_tokens must be between 100 and 1200'

            assert self.prompt_template or self.prompt_template_path, 'either prompt_template or prompt_template_name must be provided in the config'

            assert self.image_quality in ('auto', 'high', 'low'), 'config image_quality must be one of: auto, high, low'
        except AssertionError as e:
            raise ValueError(e)

        return self

class ConfigApi(pydantic.BaseModel):
    url: str = ''
    token: str = ''
    model_name: str = ''

    @pydantic.model_validator(mode='after')
    def validate_(self):
        try:
            assert self.url, 'api url must be provided'
            assert self.url.startswith('http://') or self.url.startswith('https://'), 'api url must be an http link'

            assert self.model_name, 'api model_name must be provided'
        except AssertionError as e:
            raise ValueError(e)

        return self

class ConfigDataset(pydantic.BaseModel):
    paths: list[str] = []
    images: list[DatasetImage] = []
