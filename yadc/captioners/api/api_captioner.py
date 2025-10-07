from typing import Generator

import requests
import pydantic

from enum import Enum
from urllib.parse import urlparse

from yadc.core import DatasetImage

from .base import BaseAPICaptioner
from .openai import OpenAICaptioner
from .openrouter import OpenRouterCaptioner
from .gemini import GeminiCaptioner
from .koboldcpp import KoboldcppCaptioner
from .llamacpp import LlamacppCaptioner
from .ollama import OllamaCaptioner
from .vllm import VllmCaptioner

from .types import KoboldServiceInfoResponse, OpenAIModelsResponse

OPENAI_DOMAIN = 'api.openai.com'
OPENROUTER_DOMAIN = 'openrouter.ai'
GEMINI_DOMAIN = 'generativelanguage.googleapis.com'
VORTEX_DOMAIN = '-aiplatform.googleapis.com'

class APITypes(str, Enum):
    GEMINI = 'gemini'
    OPENAI = 'openai'
    OPENROUTER = 'openrouter'
    LLAMACPP = 'llamacpp'
    KOBOLDCPP = 'koboldcpp'
    VLLM = 'vllm'
    OLLAMA = 'ollama'

    def __str__(self) -> str:
        return self.value

class APICaptioner(BaseAPICaptioner):
    api_type: APITypes
    inner_captioner: BaseAPICaptioner

    def __init__(self, **kwargs):
        """
        Initializes the APICaptioner with API and template configuration.

        Args:
            api_url (str): Base URL for the API endpoint.
            api_token (str): API key for authentication. (optional for local backend)

            **kwargs: Optional keyword arguments:
                - `prompt_template` (str): The prompt template used for captioning. If none is provided, the default will be used.
                - `image_quality` (str): Quality setting for encoded images ('auto', 'low', 'high').
                - `reasoning` (bool): Enable internal chain-of-thought / extra reasoning behavior.
                - `reasoning_effort` (str, optional): Level of reasoning effort to request when `reasoning` is True ('low', 'medium', 'high'). 
                - `reasoning_exclude_output` (bool, optional): When True, exclude internal reasoning output from the caption.
                - `session` (requests.Session, options): Override the session for the API calls
                - `cache` (yadc.captioners.utils.cache.HTTPResponseCache, options): Sets the session cache

        Raises:
            ValueError: If `api_url` is not provided.
        """

        super().__init__(**kwargs)
        kwargs['_warnings'] = False

        try:
            self.api_type = self._infer_api_type()
        except requests.ConnectionError:
            raise ValueError(f'failed to infer captioner by api url ({self._api_url}): api is down')
        except Exception as e:
            raise ValueError(f'failed to infer captioner by api url ({self._api_url}): are you using the wrong url? (e.g. http://localhost:5001/v1): {e}') from e

        match self.api_type:
            case APITypes.OPENAI:
                self.inner_captioner = OpenAICaptioner(**kwargs)

            case APITypes.OPENROUTER:
                self.inner_captioner = OpenRouterCaptioner(**kwargs)

            case APITypes.GEMINI:
                self.inner_captioner = GeminiCaptioner(**kwargs)

        # rest are uncached
        kwargs.pop('cache', None)

        match self.api_type:
            case APITypes.LLAMACPP:
                self.inner_captioner = LlamacppCaptioner(**kwargs)

            case APITypes.KOBOLDCPP:
                self.inner_captioner = KoboldcppCaptioner(**kwargs)

            case APITypes.VLLM:
                self.inner_captioner = VllmCaptioner(**kwargs)

            case APITypes.OLLAMA:
                self.inner_captioner = OllamaCaptioner(**kwargs)

        assert hasattr(self, 'inner_captioner')

    def _infer_api_type(self):
        # early exit
        try:
            api_url = urlparse(self._api_url)

            if api_url.netloc == OPENAI_DOMAIN:
                return APITypes.OPENAI

            if api_url.netloc == OPENROUTER_DOMAIN:
                return APITypes.OPENROUTER

            if api_url.netloc == GEMINI_DOMAIN:
                return APITypes.GEMINI

            if api_url.netloc.endswith(VORTEX_DOMAIN):
                return APITypes.GEMINI
        except:
            pass

        # infer based on models response initially

        # llamacpp: owned_by set to 'llamacpp'
        # kobold: models are prefixed with 'koboldcpp/'
        #         and owned_by set to 'koboldcpp'
        # vllm: owned_by set to 'vllm'
        # ollama: owned_by set to ollama user or 'library'

        try:
            with self._session.get('models') as models_resp:
                assert models_resp.ok, f'request failed with http {models_resp.status_code}'

                models_json = models_resp.json()
                assert isinstance(models_json, dict), f'bad models response type; expected dict, got {type(models_json)}'

                models = OpenAIModelsResponse(**models_json)

                for model in models.data:
                    if model.owned_by == 'llamacpp':
                        return APITypes.LLAMACPP

                    if model.owned_by == 'koboldcpp':
                        return APITypes.KOBOLDCPP

                    if model.owned_by == 'vllm':
                        return APITypes.VLLM
        except AssertionError as e:
            raise ValueError(f'failed to retrieve model list: {e}')


        # fallback to specific api checks

        try:
            with self._session.get('/health') as health_resp:
                assert health_resp.ok

                server = health_resp.headers.get('server', 'unknown').lower()

                if 'llama.cpp' in server:
                    return APITypes.LLAMACPP
        except:
            pass

        try:
            # koboldcpp has well defined endpoint
            #
            # reference: https://lite.koboldai.net/koboldcpp_api#/serviceinfo/get__well_known_serviceinfo

            with self._session.get('/.well-known/serviceinfo') as koboldcpp_resp:
                assert koboldcpp_resp.ok

                koboldcpp_json = koboldcpp_resp.json()
                assert isinstance(koboldcpp_json, dict)

                koboldcpp_service_info = KoboldServiceInfoResponse(**koboldcpp_json)

                assert koboldcpp_service_info.software.name.lower() == 'koboldcpp'

                return APITypes.KOBOLDCPP
        except AssertionError:
            pass
        except requests.JSONDecodeError:
            pass
        except pydantic.ValidationError:
            pass

        try:
            # reference: https://github.com/ollama/ollama/blob/c23e6f4cae3cbf62db68c2c9bf993925626fbe7c/server/routes.go#L1413
            # HEAD / returns "Ollama is running"
            with self._session.request('HEAD', '/') as models_resp:
                assert models_resp.ok

                ollama_resp_text = models_resp.text.lower()
                if 'ollama' in ollama_resp_text:
                    return APITypes.OLLAMA
        except AssertionError:
            pass


        # last fallback
        # the following aren't very good checks (might not work in the future)

        try:
            # reference: https://github.com/ollama/ollama/blob/main/docs/api.md#version
            # GET /api/version will return the ollama version
            # this is not really a good check, since this might be the same as other software

            with self._session.get('/api/version') as models_resp:
                assert models_resp.ok

                ollama_json = models_resp.json()
                assert isinstance(ollama_json, dict)
                assert 'version' in ollama_json

            return APITypes.OLLAMA
        except AssertionError:
            pass
        except requests.JSONDecodeError:
            pass


        try:
            # vllm doesn't list their apis clearly
            # so, check for some found in their code
            #
            # /ping: https://github.com/vllm-project/vllm/blob/9fac6aa30b669de75d8718164cd99676d3530e7d/vllm/entrypoints/openai/api_server.py#L365
            # /version: https://github.com/vllm-project/vllm/blob/9fac6aa30b669de75d8718164cd99676d3530e7d/vllm/entrypoints/openai/api_server.py#L465

            with self._session.get('/ping') as vllm_resp:
                assert vllm_resp.ok

            with self._session.post('/ping') as vllm_resp:
                assert vllm_resp.ok

            with self._session.get('/version') as vllm_resp:
                assert vllm_resp.ok

                vllm_json = vllm_resp.json()
                assert isinstance(vllm_json, dict)
                assert 'version' in vllm_json

            return APITypes.VLLM
        except AssertionError:
            pass
        except requests.JSONDecodeError:
            pass

        return APITypes.OPENAI


    def log_usage(self):
        self.inner_captioner.log_usage()

    def load_model(self, model_repo: str, **kwargs) -> None:
        return self.inner_captioner.load_model(model_repo, **kwargs)

    def unload_model(self) -> None:
        self.inner_captioner.unload_model()

    def offload_model(self) -> None:
        self.inner_captioner.offload_model()

    def predict_stream(self, image: DatasetImage, **kwargs) -> 'Generator[str, None, None]':
        return self.inner_captioner.predict_stream(image, **kwargs)

    def predict(self, image: DatasetImage, **kwargs) -> str:
        return self.inner_captioner.predict(image, **kwargs)
