from transformers import Gemma3nForConditionalGeneration

from yadc.core import logging

from .base import BaseTransformersCaptioner

GEMMA_REPOS = [
    "google/gemma-3n-E2B-it",
    "google/gemma-3n-E4B-it",
]

GEMMA_MODULE_COMPILATION = {
    'Gemma3nModel': {
        'mode': 'default',
        'dynamic': True,
        'fullgraph': True,
    },
}

_logger = logging.get_logger(__name__)

class Gemma3nCaptioner(BaseTransformersCaptioner):
    def __init__(self, **kwargs):
        kwargs['_model_name'] = 'Gemma 3n'
        kwargs['_model_repo_list'] = GEMMA_REPOS
        kwargs['_module_compilation'] = GEMMA_MODULE_COMPILATION

        super().__init__(**kwargs)

    def _from_pretrained(self, model_repo: str, **kwargs):
        return Gemma3nForConditionalGeneration.from_pretrained(
            model_repo,
            device_map="auto",
            dtype="auto",
            **kwargs,
        )
