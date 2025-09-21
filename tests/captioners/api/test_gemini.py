import re
import mock
import pytest

import requests
import requests_mock

from yadc.core import DatasetImage
from yadc.captioners.api import APICaptioner

@pytest.fixture
def gemini(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _gemini(case: str, model: str, base_url: str = 'mock://generativelanguage.googleapis.com/v1beta'):
        request_mocker.register_uri('GET', f'{base_url}/models/unknown', status_code=404)

        request_mocker.register_uri('GET', f'{base_url}/models/{model}', json={'name': model, 'version': '1', 'displayName': 'Model', 'supportedGenerationMethods': ['generateContent'], 'thinking': True})
        request_mocker.register_uri('GET', f'{base_url}/models', json={'models': [{'name': model, 'version': '1', 'displayName': 'Model', 'supportedGenerationMethods': ['generateContent'], 'thinking': True}]})

        request_mocker.register_uri('POST', f'{base_url}/models/{model}:streamGenerateContent?alt=sse', text=load_test_data(case))

        captioner = APICaptioner(
            api_url=base_url,
            api_token='secret token',
            session=session,
        )

        return captioner

    return _gemini

def test_gemini(gemini, load_test_data):
    captioner: APICaptioner = gemini('gemini.txt', 'gemini-2.5-flash')
    captioner.load_model('gemini-2.5-flash')

    expected = load_test_data('gemini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_gemini_raises_error_on_bad_model(gemini):
    captioner: APICaptioner = gemini('gemini.txt', 'gemini-2.5-flash')

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('unknown')
