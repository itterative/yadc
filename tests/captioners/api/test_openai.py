
import re
import mock
import pytest

import requests
import requests_mock

from yadc.core import DatasetImage
from yadc.captioners.api import APICaptioner

@pytest.fixture
def openai(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _openrouter(case: str, model: str, base_url: str = 'mock://api.openai.com/v1'):
        request_mocker.register_uri('GET', f'{base_url}/models', json={'data': [{'id': model, 'object': 'model', 'owned_by': 'openai'}]})

        request_mocker.register_uri('POST', f'{base_url}/chat/completions', text=load_test_data(case))

        captioner = APICaptioner(api_url=f'{base_url}', api_token='api_token', session=session)

        return captioner

    return _openrouter

def test_openai_o4_mini(openai, load_test_data):
    captioner: APICaptioner = openai('openai_o4_mini.txt', 'o4-mini')
    captioner.load_model('o4-mini')

    expected = load_test_data('openai_o4_mini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_openai_raises_error_on_bad_model(openai):
    captioner: APICaptioner = openai('openai_o4_mini.txt', 'o4-mini')

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('unknown')
