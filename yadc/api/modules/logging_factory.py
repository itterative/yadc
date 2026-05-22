import logging

from ..configuration import Configuration
from .service import Service


class LoggingFactory(Service):
    def __init__(self, configuration: Configuration) -> None:
        self.log_level: int = configuration.logging_default_level
        self._loggers: dict[str, logging.Logger] = {}

        logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

    def set_level(self, level: int):
        self.log_level = level

        for logger in self._loggers.values():
            logger.setLevel(level)

    def get_logger(self, name: str):
        logger = self._loggers.get(name, None)

        if logger is not None:
            return logger

        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        self._loggers[name] = logger
        return logger
