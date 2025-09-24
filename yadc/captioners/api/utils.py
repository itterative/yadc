from typing import Generator

from yadc.core import logging
from yadc.core.utils import Timer

_logger = logging.get_logger(__name__)

FLAG_STRIP_CAPTION = True

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
            raw_buffer = raw_buffer.lstrip() # just in case, just remove starting whitespace

            # edge case: somehow kimi-vl returns this instead of angled brackets
            content = content.replace('◁', '<')
            content = content.replace('▷', '>')

            buffer += content
            buffer = buffer.lstrip() # just in case, just remove starting whitespace

            if not buffer.startswith('<'):
                buffer = ''
                break

            if len(buffer) < len(thinking_start):
                continue

            if not buffer.startswith(thinking_start):
                buffer = ''
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
            raw_buffer = raw_buffer[i_thinking_end+len(thinking_end):]

            buffer = ''
            break

    if thinking_buffer:
        _logger.debug(thinking_buffer)

    if did_think:
        _logger.info('Thinking done (%.2f sec)\n', timer.elapsed)

    if not FLAG_STRIP_CAPTION:
        yield raw_buffer
        yield from stream
        return

    buffer = raw_buffer

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

def handle_thinking(content: str):
    thinking_start = '<think>'
    thinking_end = '</think>'

    thinking_buffer = ''

    content = content.lstrip() # just in case, just remove starting whitespace
    raw_content = content

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

    if not FLAG_STRIP_CAPTION:
        return content

    return content.strip()
