from typing import Any

import json
import pydantic
import requests

from yadc.core import logging

from .types import (
    OpenAIErrorResponse,
    OpenAIStreamingResponse,
    OpenRouterModerationError,
    GeminiErrorResponse,
    GeminiStreamingResponse,
)

_logger = logging.get_logger(__name__)

class _ParsedError:
    def __init__(self, error_source: str, error_code: int, error_message: str):
        self.source = error_source
        self.code = error_code
        self.message = error_message

class ErrorNormalizationMixin:
    class GenerationError(BaseException):
        def __init__(self, error: str) -> None:
            super().__init__(error)

    def _normalize_error(self, error: Any):
        def _try_parse_moderation(moderation: dict):
            try:
                moderation_error = OpenRouterModerationError(**moderation)
                return _ParsedError('moderation', 400, '; '.join(moderation_error.reasons))
            except:
                pass

            return None

        def _try_parse_error_json(error_source: str, error_text: str):
            try:
                assert error_text

                error_json = json.loads(error_text)
                assert isinstance(error_json, dict)
            except:
                return None

            try:
                error_response = OpenAIErrorResponse(**error_json).error

                error_code = error_response.code
                error_message = error_response.message

                if error_response.metadata and (error_parsed := _try_parse_moderation(error_response.metadata)):
                    return error_parsed

                return _ParsedError(error_source, error_code, error_message)
            except:
                pass

            try:
                error_response = GeminiErrorResponse(**error_json).error

                error_code = error_response.code
                error_message = error_response.message

                return _ParsedError(error_source, error_code, f'({error_response.status}) {error_message}')
            except:
                pass

            return None

        error_source = 'app'
        error_code = -1
        error_message = ''

        if isinstance(error, requests.HTTPError):
            _logger.debug('HTTP error %d: %s', error.response.status_code, error.response.text)

            error_source = 'http'
            error_code = error.response.status_code
            error_message = ''

            if error_parsed := _try_parse_error_json(error_source, error.response.text):
                return f'api returned an error ({error_parsed.source} {error_parsed.code}): {error_parsed.message}'

            _logger.warning('Warning: failed to process http error: %d', error.response.status_code)
        elif isinstance(error, ErrorNormalizationMixin.GenerationError):
            _logger.debug('Generation error: %s', error)

            error_source = 'generation'
            error_code = 500

            if error_parsed := _try_parse_error_json(error_source, str(error)):
                return f'api returned an error ({error_parsed.source} {error_parsed.code}): {error_parsed.message}'

            _logger.warning('Warning: failed to process generation error')
        elif isinstance(error, OpenAIStreamingResponse):
            error_response = error.error

            if error_response is not None:
                if error_response.metadata and (error_parsed := _try_parse_moderation(error_response.metadata)):
                    return error_parsed

            for choice in error.choices:
                if not choice.finish_reason:
                    continue

                return f'api stopped generating: reason: {choice.finish_reason}'

            _logger.warning('Warning: failed to process openai streaming response')
        elif isinstance(error, GeminiStreamingResponse):
            if error.promptFeedback:
                return f'api stopped generating: reason: {error.promptFeedback.blockReason}: {"; ".join(error.promptFeedback.safetyRatings)}'

            for candidate in error.candidates:
                if not candidate.finishReason:
                    continue

                return f'api stopped generating: reason: {candidate.finishReason}'

            _logger.warning('Warning: failed to process gemini streaming response')
        else:
            _logger.debug('Unhandled error %s: %s', type(error), error)

            return f'unknown error: {type(error)}'

        # some defaults if response cannot be processed
        if not error_message:
            match (error_source, error_code):
                case ('http', 400): error_message = 'request could not be completed'
                case ('http', 401): error_message = 'authentication failure'
                case ('http', 402): error_message = 'not enough api credits; payment needed'
                case ('http', 404): error_message = 'model not found'
                case ('http', 408): error_message = 'timeout'
                case ('http', 429): error_message = 'overloaded'
                case ('http', 502): error_message = 'unavailable'
                case ('http', 503): error_message = 'unavailable'
                case ('app', _): error_message = 'unknown error'
    
        return f'api returned an error ({error_source} {error_code}): {error_message}'
