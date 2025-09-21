
from yadc.core import logging
from yadc.core import DatasetImage

from .openai import OpenAICaptioner, APITypes

_logger = logging.get_logger(__name__)

class VllmCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs):
        self._api_type = APITypes.VLLM
        super().__init__(**kwargs)


    def conversation(self, image: DatasetImage, **kwargs):
        conversation = super().conversation(image, **kwargs)

        conversation['max_tokens'] = conversation.pop('max_completion_tokens', 512)

        return conversation
