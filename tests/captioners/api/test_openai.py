
import re
import mock
import pytest

import requests
import requests_mock

from yadc.core import DatasetImage
from yadc.captioners.api import OpenAICaptioner

@pytest.fixture
def koboldcpp(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _koboldcpp(case: str, model: str, loaded: bool = True, base_url: str = 'mock://koboldcpp'):
        loaded_model = model if loaded else 'inactive'

        def load_model(r: requests.Request, c):
            nonlocal loaded_model

            data = r.json() # type: ignore
            assert isinstance(data, dict)

            data_filename = data.get('filename')
            assert isinstance(data_filename, str)

            loaded_model = 'inactive' if data_filename == 'unload_model' else data_filename
            return {'success': True}

        request_mocker.register_uri('GET', f'{base_url}/.well-known/serviceinfo', json={'software': {'name': 'koboldcpp'}})
        request_mocker.register_uri('GET', f'{base_url}/v1/models', json=lambda r, c: {'data': [{'id': loaded_model, 'object': 'model', 'owned_by': 'koboldcpp'}]})
        request_mocker.register_uri('GET', f'{base_url}/api/v1/model', json=lambda r, c: {'result': loaded_model})
        request_mocker.register_uri('GET', f'{base_url}/api/admin/list_options', json=['unload_model', model])

        request_mocker.register_uri('POST', f'{base_url}/api/admin/reload_config', json=load_model)

        request_mocker.register_uri('POST', f'{base_url}/v1/chat/completions', text=load_test_data(case))

        captioner = OpenAICaptioner(api_url=f'{base_url}/v1', session=session)

        return captioner

    return _koboldcpp

@pytest.fixture
def openrouter(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _openrouter(case: str, model: str, base_url: str = 'mock://openrouter.ai/api/v1'):
        request_mocker.register_uri('GET', f'{base_url}/credits', json={'data': {'total_credits': 1.0, 'total_usage': 0.1}})
        request_mocker.register_uri('GET', f'{base_url}/models', json={'data': [{'id': model, 'object': 'model', 'owned_by': 'openrouter'}]})

        request_mocker.register_uri('POST', f'{base_url}/chat/completions', text=load_test_data(case))

        captioner = OpenAICaptioner(api_url=f'{base_url}', session=session)

        return captioner

    return _openrouter

@pytest.fixture
def openai(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _openrouter(case: str, model: str, base_url: str = 'mock://api.openai.com/v1'):
        request_mocker.register_uri('GET', f'{base_url}/models', json={'data': [{'id': model, 'object': 'model', 'owned_by': 'openai'}]})

        request_mocker.register_uri('POST', f'{base_url}/chat/completions', text=load_test_data(case))

        captioner = OpenAICaptioner(api_url=f'{base_url}', session=session)

        return captioner

    return _openrouter

def test_koboldcpp(koboldcpp, load_test_data):
    captioner: OpenAICaptioner = koboldcpp('koboldcpp.txt', 'koboldcpp/gemma-3-27b')
    captioner.load_model('koboldcpp/gemma-3-27b')

    expected = load_test_data('koboldcpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_koboldcpp_should_load_model(koboldcpp, load_test_data):
    captioner: OpenAICaptioner = koboldcpp('koboldcpp.txt', 'koboldcpp/gemma-3-27b', loaded=False)
    captioner.load_model('koboldcpp/gemma-3-27b')

    expected = load_test_data('koboldcpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_koboldcpp_should_raise_error_on_bad_model(koboldcpp, load_test_data):
    captioner: OpenAICaptioner = koboldcpp('koboldcpp.txt', 'koboldcpp/gemma-3-27b', loaded=False)

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('koboldcpp/unknown')

def test_openrouter_gpt_5_mini(openrouter, load_test_data):
    captioner: OpenAICaptioner = openrouter('openrouter_gpt_5_mini.txt', 'openai/gpt-5-mini')
    captioner.load_model('openai/gpt-5-mini')

    expected = load_test_data('openrouter_gpt_5_mini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_openrouter_raises_error_on_bad_model(openrouter, load_test_data):
    captioner: OpenAICaptioner = openrouter('openrouter_gpt_5_mini.txt', 'openai/gpt-5-mini')

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('unknown')

def test_openai_o4_mini(openai, load_test_data):
    captioner: OpenAICaptioner = openai('openai_o4_mini.txt', 'o4-mini')
    captioner.load_model('o4-mini')

    expected = load_test_data('openai_o4_mini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_openai_raises_error_on_bad_model(openai, load_test_data):
    captioner: OpenAICaptioner = openai('openai_o4_mini.txt', 'o4-mini')

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('unknown')
