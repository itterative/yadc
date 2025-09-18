import time
import json
import requests

import io
import base64

from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from yadc.core import Captioner, DatasetImage

from .session import Session
from .types import OpenAIModelsResponse, KoboldAdminSettingsReponse, KoboldAdminReloadModelReponse, KoboldAdminCurrentModelResponse

class APITypes(str, Enum):
    OPENAI = 'openai'
    KOBOLDCPP = 'koboldcpp'

    def __str__(self) -> str:
        return self.value

class OpenAICaptioner(Captioner):
    _api_type: APITypes|None = None
    _current_model: str|None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._api_url: str = kwargs.pop('api_url', '')
        self._api_token: str = kwargs.pop('api_token', '')
        self._store_conversation: bool = kwargs.pop('store_conversation', False)
        self._image_quality: str = kwargs.pop('image_quality', 'auto')

        if not self._api_url:
            raise ValueError("no api_url")

        try:
            url = urlparse(self._api_url)
            self._api_url = f'{url.scheme}://{url.netloc}'
        except:
            pass

        session_headers = {}

        if self._api_token:
            session_headers['Authorization'] = f'Bearer {self._api_token}'
        else:
            print(f'Warning: no api_token is set, requests will fail if api uses authentication')

        self._session = Session(self._api_url, headers=session_headers)

    def _infer_api_type(self):
        # early exit
        if self._api_url.startswith('https://api.openai.com'):
            return APITypes.OPENAI

        try:
            with self._session.get('/.well-known/serviceinfo') as koboldcpp_resp:
                assert koboldcpp_resp.ok
            
                koboldcpp_json = koboldcpp_resp.json()
                assert isinstance(koboldcpp_json, dict)
                assert 'software' in koboldcpp_json

                koboldcpp_sofware = koboldcpp_json['software']
                assert isinstance(koboldcpp_sofware, dict)
                assert 'name' in koboldcpp_sofware

                koboldcpp_sofware_name = koboldcpp_sofware['name']
                assert isinstance(koboldcpp_sofware_name, str)
                assert koboldcpp_sofware_name.lower() == 'koboldcpp'

                return APITypes.KOBOLDCPP
        except requests.exceptions.ConnectionError:
            raise
        except AssertionError:
            raise
        except Exception as e:
            pass

        return APITypes.OPENAI


    def load_model(self, model_repo: str, **kwargs) -> None:
        try:
            self._api_type = self._infer_api_type()
            print(f'API set to {self._api_type}.')

            match self._api_type:
                case APITypes.OPENAI:
                    self._load_model_openai(model_repo)

                case APITypes.KOBOLDCPP:
                    self._load_model_koboldcpp(model_repo)

                case _:
                    raise ValueError(f'unknown api type: {self._api_type}')
        except requests.exceptions.ConnectionError:
            raise ValueError(f'api unavailable: {self._api_url}')
        
        print(f'Model set to {self._current_model}.')

    def _load_model_openai(self, model_repo: str):
        if self._current_model == model_repo:
            return

        with self._session.get('/v1/models') as model_resp:
            assert model_resp.ok

            model_resp_json = model_resp.json()
            assert isinstance(model_resp_json, dict)

            models = OpenAIModelsResponse(**model_resp_json)
            available_models: list[str] = []

            for model in models.data:
                available_models.append(model.id)

                if model.id == model_repo:
                    self._current_model = model_repo

            if not self._current_model:
                if available_models:
                    raise ValueError(f'model not found: {model_repo}; available models: {", ".join(available_models)}')

                raise ValueError(f'model not found: {model_repo}; no models available')
            

    def _load_model_koboldcpp(self, model_repo: str, timeout: float = 60):
        if self._current_model == model_repo:
            return
        
        # koboldcpp api doesn't expose enough information to make sure that the right settings are already loaded
        # so we have to add some extra prefixes (e.g. koboldcpp/MODEL) and suffixes (e.g. MODEL.kcpps; assumes there is a matching kcpps file)

        # the logic goes as following
        #   1. if the openai endpoint shows the model is already loaded, use that
        #   2. if the kobold endpoint shows the model is already loaded, use that
        #   3. otherwise, unload the model, then wait for it to be loaded

        try:
            # early exit if the model is already loaded
            self._load_model_openai(model_repo)
            return
        except ValueError:
            pass

        model_koboldcpp = 'koboldcpp/' + model_repo
        model_kcpss = model_repo + '.kcpps'

        # early exit if already loaded
        with self._session.get('/api/v1/model') as model_current_resp:
            assert model_current_resp.ok

            model_current_resp_json = model_current_resp.json()
            assert isinstance(model_current_resp_json, dict)

            model_current = KoboldAdminCurrentModelResponse(**model_current_resp_json)

            if model_current.result == model_repo or model_current.result == model_koboldcpp:
                self._current_model = model_current.result
                return

        with self._session.get('/api/admin/list_options') as model_options_resp:
            assert model_options_resp.ok

            model_options_resp_json = model_options_resp.json()
            assert isinstance(model_options_resp_json, list)

            models = KoboldAdminSettingsReponse(data=model_options_resp_json)
            available_models: list[str] = []

            for model in models.data:
                if model == "unload_model":
                    continue

                available_models.append(model)

                if model == model_repo or model == model_kcpss:
                    self._current_model = model

            if not self._current_model:
                if available_models:
                    raise ValueError(f'model not found: {model_repo}; available models: {", ".join(available_models)}')

                raise ValueError(f'model not found: {model_repo}; no models available')

        with self._session.post('/api/admin/reload_config', json={"filename": self._current_model}) as model_reload_resp:
            assert model_reload_resp.ok

            model_reload_resp_json = model_reload_resp.json()
            assert isinstance(model_reload_resp_json, dict)

            if not KoboldAdminReloadModelReponse(**model_reload_resp_json).success:
                raise ValueError(f'failed to load model: {model_repo}')

        start_t = time.time()
        end_t = start_t + timeout

        time.sleep(5) # NOTE: the api should be down within 5s; after that we keep checking which model is loaded

        while time.time() < end_t:
            try:
                with self._session.get('/api/v1/model') as model_current_resp:
                    assert model_current_resp.ok

                    model_current_resp_json = model_current_resp.json()
                    assert isinstance(model_current_resp_json, dict)

                    model_current = KoboldAdminCurrentModelResponse(**model_current_resp_json)

                    if model_current.result == "inactive":
                        time.sleep(0.5)
                        continue

                    self._current_model = model_current.result
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
                continue
        else:
            raise TimeoutError(f'failed to load model in time: {model_repo}')


    def unload_model(self):
        match self._api_type:
            case APITypes.OPENAI:
                pass

            case APITypes.KOBOLDCPP:
                with self._session.post('/api/admin/reload_config', json={"filename": "unload_model"}) as unload_model_resp:
                    assert unload_model_resp.ok

            case _:
                raise ValueError(f'unknown api type: {self._api_type}')

    def offload_model(self):
        pass

    def conversation(self, image: DatasetImage, **kwargs):
        system_prompt, user_prompt = self._prompts_from_image(image, **kwargs)

        buffer = io.BytesIO()
        image.read_image().save(buffer, format="PNG")

        match self._api_type:
            case APITypes.KOBOLDCPP:
                pass

            case APITypes.OPENAI:
                pass

            case _:
                raise ValueError(f'unknown api type: {self._api_type}')

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}",
                        },
                        "detail": self._image_quality,
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

    def _generate_prediction(self, image: DatasetImage, **kwargs):
        assert self._current_model, "model not loaded"

        temperature = kwargs.pop('temperature', 0.8)
        top_p = kwargs.pop('top_p', 0.9)
        top_k = kwargs.pop('top_k', 64)
        max_tokens = kwargs.pop('max_new_tokens', 512)

        conversation: dict = dict(
            model=self._current_model,
            metadata={ 'topic': 'yadt:caption' },
            messages=self.conversation(image, **kwargs),
            temperature=temperature,
            stream=True,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            store=self._store_conversation,
        )

        # reference: https://github.com/LostRuins/koboldcpp/blob/575eb4095095939b016dc2e1957643ffb2dbf086/tools/server/bench/script.js#L98
        if self._api_type == APITypes.KOBOLDCPP:
            conversation['stop'] = ['<|im_end|>']

        with self._session.post('/v1/chat/completions', stream=True, json=conversation) as converation_resp:
            converation_resp.raise_for_status()

            converation_stopped = False

            for line in converation_resp.iter_lines():
                assert isinstance(line, bytes)

                if not line or converation_stopped:
                    continue

                try:
                    line = line.decode('utf-8')

                    if line.startswith('data:'):
                        line = line[len('data:'):].lstrip()

                    if line == '[DONE]':
                        converation_stopped = True
                        continue

                    line_json = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f'Warning: failed to decode line: {line}')
                    continue

                try:
                    assert isinstance(line_json, dict), "not a dict"
                    assert 'choices' in line_json, "no choices"

                    line_json_choices = line_json['choices']
                    assert isinstance(line_json_choices, list), "bad choices"

                    for choice in line_json_choices:
                        assert isinstance(choice, dict), "bad choice"
                        assert 'delta' in choice, "no delta"

                        choice_delta = choice['delta']
                        assert isinstance(choice_delta, dict), "bad choice delta"

                        if content := choice_delta.get('content', ''):
                            assert isinstance(content, str), "bad content"
                            yield content
                except AssertionError as e:
                    print(f'Error: failed to process line: {e}: {line}')
                    break


    def predict(self, image: DatasetImage, **kwargs):
        return ''.join(list(self._generate_prediction(image, **kwargs)))
    
    def predict_stream(self, image: DatasetImage, **kwargs):
        yield from self._generate_prediction(image, **kwargs)
    
