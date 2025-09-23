from yadc.core import logging
from yadc.core import DatasetImage

from . import utils
from .openai import OpenAICaptioner, APITypes

_logger = logging.get_logger(__name__)

class OllamaCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs):
        self._api_type = APITypes.OLLAMA
        super().__init__(**kwargs)


    def conversation(self, image: DatasetImage, stream: bool = False, **kwargs):
        conversation = super().conversation(image, stream=stream, **kwargs)

        conversation['max_tokens'] = conversation.pop('max_completion_tokens', 512)

        return conversation

    def predict_stream(self, image: DatasetImage, **kwargs):
        return utils.handle_thinking_streaming(super().predict_stream(image, **kwargs))
    
    def predict(self, image: DatasetImage, **kwargs):
        return utils.handle_thinking(super().predict(image, **kwargs))
