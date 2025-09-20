import abc

from yadc.core import Captioner

class BaseAPICaptioner(Captioner, abc.ABC):
    @abc.abstractmethod
    def log_usage(self):
        raise NotImplemented
