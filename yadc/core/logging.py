import logging

_default_level = 'INFO'
_default_handler: logging.Handler = logging.StreamHandler()
_default_handler.setFormatter(logging.Formatter('%(message)s'))

_loggers: dict[str, '_logger'] = {}

TRACE_LEVEL = 5

class _logger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self.handlers = logger.handlers

    def trace(self, msg, *args, **kwargs):
        self._logger.log(TRACE_LEVEL, msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._logger.log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._logger.log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._logger.log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(logging.WARNING, msg, *args, **kwargs)
    
    def addHandler(self, handler: logging.Handler):
        self._logger.addHandler(handler)

    def setLevel(self, level: str):
        self._logger.setLevel(level)


def get_logger(name: str):
    logger = _loggers.get(name, None)
    if logger is not None:
        return logger

    logger = _logger(logging.getLogger(name))
    logger.handlers.clear()
    logger.addHandler(_default_handler)

    _loggers[name] = logger

    return logger

def set_level(level: str):
    global _default_level

    level = level.upper()

    if level not in ('TRACE', 'INFO', 'WARNING', 'ERROR', 'DEBUG'):
        raise ValueError(f'invalid logging level: {level}')

    if level == 'TRACE':
        logging.addLevelName(TRACE_LEVEL, 'TRACE')

    _default_level = level
    for logger in _loggers.values():
        logger.setLevel(level)

def set_handler(handler: logging.Handler):
    global _default_handler

    for logger in _loggers.values():
        logger.handlers.clear()
        logger.addHandler(handler)

    _default_handler = handler
