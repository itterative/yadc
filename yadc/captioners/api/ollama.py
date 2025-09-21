
from yadc.core import logging

from .openai import OpenAICaptioner, APITypes

_logger = logging.get_logger(__name__)

class OllamaCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs):
        self._api_type = APITypes.OLLAMA
        super().__init__(**kwargs)
