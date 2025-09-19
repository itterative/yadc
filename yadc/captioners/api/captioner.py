from typing import Generator
from urllib.parse import urlparse

from yadc.core import Captioner, DatasetImage

from .openai import OpenAICaptioner
from .gemini import GeminiCaptioner

OPENAI_DOMAIN = 'api.openai.com'
GEMINI_DOMAIN = 'generativelanguage.googleapis.com'
VORTEX_DOMAIN = '-aiplatform.googleapis.com'

class APICaptioner(Captioner):
    inner_captioner: Captioner

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        api_url: str = kwargs.get('api_url', '')

        if not api_url:
            raise ValueError("no api_url")
        
        try:
            parsed_url = urlparse(api_url)

            if parsed_url.netloc == OPENAI_DOMAIN:
                self.inner_captioner = OpenAICaptioner(**kwargs)
            elif parsed_url.netloc == GEMINI_DOMAIN:
                self.inner_captioner = GeminiCaptioner(**kwargs)
            elif parsed_url.netloc.endswith(VORTEX_DOMAIN):
                self.inner_captioner = GeminiCaptioner(**kwargs)
            else: # default
                self.inner_captioner = OpenAICaptioner(**kwargs)
        except Exception as e:
            raise ValueError(f'failed to infer caption by api url') from e
        

    def load_model(self, model_repo: str, **kwargs) -> None:
        return self.inner_captioner.load_model(model_repo, **kwargs)

    def unload_model(self) -> None:
        self.inner_captioner.unload_model()

    def offload_model(self) -> None:
        self.inner_captioner.offload_model()

    def predict_stream(self, image: DatasetImage, **kwargs) -> 'Generator[str]':
        return self.inner_captioner.predict_stream(image, **kwargs)

    def predict(self, image: DatasetImage, **kwargs) -> str:
        return self.inner_captioner.predict(image, **kwargs)
