from yadc.core import logging
from yadc.core import Captioner

_logger = logging.get_logger(__name__)

class HfTransformersCaptioner(Captioner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['_warnings'] = False

        try:
            import transformers # type: ignore
        except ImportError:
            raise ValueError('failed to initialize captioner: transformers library is missing')

        try:
            import diffusers # type: ignore
        except ImportError:
            raise ValueError('failed to initialize captioner: diffusers library is missing')

        try:
            import torch # type: ignore
        except ImportError:
            raise ValueError('failed to initialize captioner: torch library is missing')

        try:
            import torchao # type: ignore
            kwargs['_flag_torchao_disabled'] = False
        except ImportError:
            _logger.warning("Warning: torchao library is missing. Quantization using torch is disabled.")
            kwargs['_flag_torchao_disabled'] = True
