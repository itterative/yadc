from typing import Optional

from yadc.core import logging
from yadc.core import Captioner
from yadc.core.dataset import DatasetImage

_logger = logging.get_logger(__name__)

class HfTransformersCaptioner(Captioner):
    _captioner_kwargs: dict
    inner_captioner: Optional[Captioner] = None

    def __init__(self, **kwargs):
        kwargs['_warnings'] = False
        self._captioner_kwargs = kwargs

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

    def load_model(self, model_repo: str, **kwargs) -> None:
        from .gemma_3n_captioner import GEMMA_REPOS, Gemma3nCaptioner
        from .qwen3_vl_captioner import QWEN_REPOS, Qwen3VLCaptioner

        if model_repo in GEMMA_REPOS:
            self.inner_captioner = Gemma3nCaptioner(self._captioner_kwargs)
        elif model_repo in QWEN_REPOS:
            self.inner_captioner = Qwen3VLCaptioner(self._captioner_kwargs)
        else:
            raise ValueError(f'model repo not supported: {model_repo}')

        return self.inner_captioner.load_model(model_repo, **kwargs)

    def unload_model(self):
        assert self.inner_captioner is not None
        self.inner_captioner.unload_model()

    def offload_model(self):
        assert self.inner_captioner is not None
        self.inner_captioner.offload_model()

    def predict_stream(self, image: DatasetImage, **kwargs):
        assert self.inner_captioner is not None
        return self.inner_captioner.predict_stream(image, **kwargs)

    def predict(self, image: DatasetImage, **kwargs):
        assert self.inner_captioner is not None
        return self.inner_captioner.predict(image, **kwargs)
