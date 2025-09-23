from typing import Generator

from yadc.core import logging
from yadc.core.utils import Timer

_logger = logging.get_logger(__name__)

def handle_thinking_streaming(stream: Generator[str]):
    thinking_start = '<think>'
    thinking_end = '</think>'

    is_thinking = False
    did_think = False

    thinking_buffer = ''
    raw_buffer = ''
    buffer = ''

    with Timer() as timer:
        for content in stream:
            raw_buffer += content # no processing

            # edge case: somehow kimi-vl returns this instead of angled brackets
            content = content.replace('◁', '<')
            content = content.replace('▷', '>')

            buffer += content
            buffer = buffer.lstrip() # just in case, just remove starting whitespace

            if not buffer.startswith('<'):
                yield raw_buffer

                buffer = ''
                raw_buffer = ''
                break

            if len(buffer) < len(thinking_start):
                continue

            if not buffer.startswith(thinking_start):
                yield raw_buffer

                buffer = ''
                raw_buffer = ''
                break

            if not is_thinking:
                _logger.debug('')
                _logger.info('Thinking...')
            is_thinking = True

            try:
                i_thinking_end = buffer.rindex(thinking_end)
            except ValueError:
                continue

            did_think = True

            # note: should be the same size as long as the edge cases don't change length
            thinking_buffer = raw_buffer[len(thinking_start):i_thinking_end]
            yield raw_buffer[i_thinking_end+len(thinking_end):]

            buffer = ''
            raw_buffer = ''
            break

    if thinking_buffer:
        _logger.debug(thinking_buffer)

    if did_think:
        _logger.info('Thinking done (%.2f sec)\n', timer.elapsed)

    yield from stream

def handle_thinking(content: str):
    thinking_start = '<think>'
    thinking_end = '</think>'

    thinking_buffer = ''
    raw_content = content

    content = content.lstrip() # just in case, just remove starting whitespace

    # edge case: somehow kimi-vl returns this instead of angled brackets
    content = content.replace('◁', '<')
    content = content.replace('▷', '>')

    if not content.startswith(thinking_start):
        return raw_content
    
    try:
        i_thinking_end = content.rindex(thinking_end)
    except ValueError:
        return raw_content

    # note: should be the same size as long as the edge cases don't change length
    thinking_buffer = raw_content[len(thinking_start):i_thinking_end]
    content = raw_content[i_thinking_end+len(thinking_end):]

    if thinking_buffer:
        _logger.debug('')
        _logger.debug('THOUGHT -------')
        _logger.debug(thinking_buffer)
        _logger.debug('---------------')
        _logger.debug('')

    return content