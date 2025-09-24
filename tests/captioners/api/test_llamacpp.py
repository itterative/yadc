
import re
import mock
import pytest

import requests
import requests_mock

from yadc.core import DatasetImage
from yadc.captioners.api import APICaptioner

@pytest.fixture
def llamacpp(session: requests.Session, request_mocker: requests_mock.Adapter, load_test_data):
    def _llamacpp(case: str, model: str, base_url: str = 'mock://llamacpp'):

        request_mocker.register_uri('GET', f'{base_url}/v1/models', json=lambda r, c: {'data': [{'id': model, 'object': 'model', 'owned_by': 'llamacpp'}]})
        request_mocker.register_uri('POST', f'{base_url}/v1/chat/completions', text=load_test_data(case))

        captioner = APICaptioner(api_url=f'{base_url}/v1', session=session)

        return captioner

    return _llamacpp

def test_llamacpp(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp('nonstreaming/llamacpp.txt', 'llamacpp/gemma-3-27b')
    captioner.load_model('llamacpp/gemma-3-27b')

    expected = load_test_data('nonstreaming/llamacpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_llamacpp_streaming(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp('streaming/llamacpp.txt', 'llamacpp/gemma-3-27b')
    captioner.load_model('llamacpp/gemma-3-27b')

    expected = load_test_data('streaming/llamacpp_result.txt')
    got = ''.join(captioner.predict_stream(mock.MagicMock(spec=DatasetImage)))

    assert got == expected, 'bad prediction'

def test_llamacpp_cot(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp('nonstreaming/llamacpp_cot.txt', 'llamacpp/gemma-3-27b')
    captioner.load_model('llamacpp/gemma-3-27b')

    expected = load_test_data('nonstreaming/llamacpp_cot_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_llamacpp_streaming_cot(llamacpp, load_test_data):
    captioner: APICaptioner = llamacpp('streaming/llamacpp_cot.txt', 'llamacpp/gemma-3-27b')
    captioner.load_model('llamacpp/gemma-3-27b')

    expected = load_test_data('streaming/llamacpp_cot_result.txt')
    got = ''.join(captioner.predict_stream(mock.MagicMock(spec=DatasetImage)))

    assert got == expected, 'bad prediction'

def test_llamacpp_should_raise_error_on_bad_model(llamacpp):
    captioner: APICaptioner = llamacpp('nonstreaming/llamacpp.txt', 'llamacpp/gemma-3-27b')

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('llamacpp/unknown')
