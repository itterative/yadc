
import time
import requests

from yadc.core import logging
from yadc.core import DatasetImage

from .types import (
    KoboldAdminSettingsReponse,
    KoboldAdminReloadModelReponse,
    KoboldAdminCurrentModelResponse,
)

from .openai import OpenAICaptioner, APITypes

_logger = logging.get_logger(__name__)

class KoboldcppCaptioner(OpenAICaptioner):
    def __init__(self, **kwargs):
        self._api_type = APITypes.KOBOLDCPP
        super().__init__(**kwargs)


    def _load_model(self, model_repo: str, timeout: float = 60):
        if self._current_model == model_repo:
            return

        # koboldcpp api doesn't expose enough information to make sure that the right settings are already loaded
        # so we have to add some extra prefixes (e.g. koboldcpp/MODEL) and suffixes (e.g. MODEL.kcpps; assumes there is a matching kcpps file)

        # the logic goes as following
        #   1. if the openai endpoint shows the model is already loaded, use that
        #   2. if the kobold endpoint shows the model is already loaded, use that
        #   3. otherwise, unload the model, then wait for it to be loaded

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

        # time.sleep(5) # NOTE: the api should be down within 5s; after that we keep checking which model is loaded

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
        try:
            with self._session.post('/api/admin/reload_config', json={"filename": "unload_model"}) as unload_model_resp:
                assert unload_model_resp.ok
        except AssertionError as e:
            raise ValueError("failed to unload model") from e


    def conversation(self, image: DatasetImage, **kwargs):
        conversation = super().conversation(image, **kwargs)

        # reference: https://github.com/LostRuins/koboldcpp/blob/575eb4095095939b016dc2e1957643ffb2dbf086/tools/server/bench/script.js#L98
        conversation.setdefault('stop', ['<|im_end|>'])

        return conversation
