import logging

_default_level = 'INFO'
_default_handler = logging.StreamHandler()
_default_handler.setFormatter(logging.Formatter('%(message)s'))

_loggers: dict[str, logging.Logger] = {}

def get_logger(name: str):
    logger = _loggers.get(name, None)
    if logger is not None:
        return logger

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(_default_handler)

    _loggers[name] = logger

    return logger

def set_level(level: str):
    global _default_level

    level = level.upper()

    if level not in ('INFO', 'WARNING', 'ERROR', 'DEBUG'):
        raise ValueError(f'invalid logging level: {level}')

    _default_level = level
    for logger in _loggers.values():
        logger.setLevel(level)
