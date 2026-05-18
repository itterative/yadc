from typing import Any

from typing_extensions import override

from yadc.core import DatasetImage, logging

from .openai import APITypes, OpenAICaptioner

_logger = logging.get_logger(__name__)


class VllmCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs: Any):
        super().__init__(api_type=APITypes.VLLM, **kwargs)

    @override
    def conversation(self, image: DatasetImage, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        conversation = super().conversation(image, stream=stream, **kwargs)

        conversation["max_tokens"] = conversation.pop("max_completion_tokens", 512)

        return conversation
