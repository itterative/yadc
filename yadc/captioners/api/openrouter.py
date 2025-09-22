from yadc.core import logging
from yadc.core import DatasetImage

from .types import OpenRouterCreditsResponse

from .openai import OpenAICaptioner, APITypes

_logger = logging.get_logger(__name__)

class OpenRouterCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs):
        self._api_type = APITypes.OPENROUTER
        super().__init__(**kwargs)

        self._log_api_information()

    def _log_api_information(self):
        with self._session.get('credits') as credits_resp:
            try:
                credits_resp.raise_for_status()

                credits_resp_json = credits_resp.json()
                assert isinstance(credits_resp_json, dict)

                credits = OpenRouterCreditsResponse(**credits_resp_json).data
                _logger.info('You have used %.2f out of %.2f credits with this api token.', credits.total_usage, credits.total_credits)
            except:
                _logger.warning('Warning: failed to retrieve current credits for api token.')

    def conversation(self, image: DatasetImage, stream: bool = False, **kwargs):
        conversation = super().conversation(image, stream=stream, **kwargs)

        if self._reasoning:
            conversation.pop('reasoning_effort', None) # remove any existing openai reasoning config

            conversation['reasoning'] = {
                'effort': self._reasoning_effort,
                'exclude': self._reasoning_exclude_output,
            }

        conversation.pop('stream_options', None)
        conversation['usage'] = { 'include': True }

        conversation['max_tokens'] = conversation.pop('max_completion_tokens', 512)

        return conversation
