
import re
import mock
import pytest

import requests
import requests_mock

from yadc.core import DatasetImage
from yadc.captioners.api import APICaptioner

@pytest.fixture
def openrouter(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _openrouter(case: str, model: str, base_url: str = 'mock://openrouter.ai/api/v1'):
        request_mocker.register_uri('GET', f'{base_url}/credits', json={'data': {'total_credits': 1.0, 'total_usage': 0.1}})
        request_mocker.register_uri('GET', f'{base_url}/models', json={'data': [{'id': model, 'object': 'model', 'owned_by': 'openrouter'}]})

        request_mocker.register_uri('POST', f'{base_url}/chat/completions', text=load_test_data(case))

        captioner = APICaptioner(api_url=f'{base_url}', session=session)

        return captioner

    return _openrouter

def test_openrouter_gpt_5_mini(openrouter, load_test_data):
    captioner: APICaptioner = openrouter('nonstreaming/openrouter_gpt_5_mini.txt', 'openai/gpt-5-mini')
    captioner.load_model('openai/gpt-5-mini')

    expected = load_test_data('nonstreaming/openrouter_gpt_5_mini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_openrouter_gpt_5_mini_streaming(openrouter, load_test_data):
    captioner: APICaptioner = openrouter('streaming/openrouter_gpt_5_mini.txt', 'openai/gpt-5-mini')
    captioner.load_model('openai/gpt-5-mini')

    expected = load_test_data('streaming/openrouter_gpt_5_mini_result.txt')
    got = ''.join(captioner.predict_stream(mock.MagicMock(spec=DatasetImage)))

    assert got == expected, 'bad prediction'

def test_openrouter_raises_error_on_bad_model(openrouter):
    captioner: APICaptioner = openrouter('nonstreaming/openrouter_gpt_5_mini.txt', 'openai/gpt-5-mini')

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('unknown')
