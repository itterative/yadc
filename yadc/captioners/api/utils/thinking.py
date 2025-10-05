from typing import Generator

from yadc.core import logging
from yadc.core.utils import Timer

_logger = logging.get_logger(__name__)

FLAG_STRIP_CAPTION = True

class ThinkingMixin:
    def __init__(self, **kwargs):
        self._reasoning_start_token: str = kwargs.pop('reasoning_start_token', '<think>')
        self._reasoning_end_token: str = kwargs.pop('reasoning_end_token', '</think>')

    def _handle_thinking_streaming(self, stream: Generator[str, None, None]):
        is_thinking = False
        did_think = False

        thinking_buffer = ''
        buffer = ''

        with Timer() as timer:
            for content in stream:
                buffer += content
                buffer = buffer.lstrip() # just in case, just remove starting whitespace

                if not buffer.startswith(self._reasoning_start_token[:1]):
                    break

                if len(buffer) < len(self._reasoning_start_token):
                    continue

                if not buffer.startswith(self._reasoning_start_token):
                    break

                if not is_thinking:
                    _logger.debug('')
                    _logger.info('Thinking...')
                is_thinking = True

                try:
                    i_thinking_end = buffer.rindex(self._reasoning_end_token)
                except ValueError:
                    continue

                did_think = True

                # note: should be the same size as long as the edge cases don't change length
                thinking_buffer = buffer[len(self._reasoning_start_token):i_thinking_end]
                buffer = buffer[i_thinking_end+len(self._reasoning_end_token):]

                buffer = ''
                break

        if is_thinking and not did_think:
            thinking_buffer = buffer.removeprefix(self._reasoning_start_token)
            _logger.info(_indent_thinking(thinking_buffer))
            _logger.warning('Warning: Thinking was not finished.')
            return

        if thinking_buffer:
            _logger.info(_indent_thinking(thinking_buffer))

        if did_think:
            _logger.info('Thinking done (%.2f sec)\n', timer.elapsed)

        if not FLAG_STRIP_CAPTION:
            yield buffer
            yield from stream
            return

        for content in stream:
            buffer += content

            if not buffer or buffer.isspace():
                continue

            yield buffer.lstrip()
            break

        buffer = ''

        for content in stream:
            buffer += content

            if not content or content.isspace():
                continue

            yield buffer
            buffer = ''

        yield buffer.rstrip()

    def _handle_thinking(self, content: str):
        thinking_buffer = ''

        content = content.lstrip() # just in case, just remove starting whitespace

        if not content.startswith(self._reasoning_start_token):
            return content

        try:
            i_thinking_end = content.rindex(self._reasoning_end_token)
        except ValueError:
            thinking_buffer = content.removeprefix(self._reasoning_start_token)
            _logger.info('Thinking...')
            _logger.info(_indent_thinking(thinking_buffer))
            _logger.warning('Warning: Thinking was not finished.')
            return ''

        # note: should be the same size as long as the edge cases don't change length
        thinking_buffer = content[len(self._reasoning_start_token):i_thinking_end]
        content = content[i_thinking_end+len(self._reasoning_end_token):]

        if thinking_buffer:
            _logger.debug('')
            _logger.info('Thinking...')
            _logger.info(_indent_thinking(thinking_buffer))
            _logger.info('Thinking done.\n')

        if not FLAG_STRIP_CAPTION:
            return content

        return content.strip()

def _indent_thinking(buffer: str):
    return '\n'.join(map(lambda s: '> ' + s, buffer.strip().splitlines()))
