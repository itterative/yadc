
from yadc.core import logging

from .openai import OpenAICaptioner, APITypes

_logger = logging.get_logger(__name__)

class VllmCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs):
        self._api_type = APITypes.VLLM
        super().__init__(**kwargs)
