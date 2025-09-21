
import re
import mock
import pytest

import requests
import requests_mock

from yadc.core import DatasetImage
from yadc.captioners.api import APICaptioner

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

        captioner = APICaptioner(api_url=f'{base_url}/v1', session=session)

        return captioner

    return _koboldcpp

def test_koboldcpp(koboldcpp, load_test_data):
    captioner: APICaptioner = koboldcpp('koboldcpp.txt', 'koboldcpp/gemma-3-27b')
    captioner.load_model('koboldcpp/gemma-3-27b')

    expected = load_test_data('koboldcpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_koboldcpp_should_load_model(koboldcpp, load_test_data):
    captioner: APICaptioner = koboldcpp('koboldcpp.txt', 'koboldcpp/gemma-3-27b', loaded=False)
    captioner.load_model('koboldcpp/gemma-3-27b')

    expected = load_test_data('koboldcpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_koboldcpp_should_raise_error_on_bad_model(koboldcpp):
    captioner: APICaptioner = koboldcpp('koboldcpp.txt', 'koboldcpp/gemma-3-27b', loaded=False)

    with pytest.raises(ValueError, match=re.compile('model not found: .*')):
        captioner.load_model('koboldcpp/unknown')
