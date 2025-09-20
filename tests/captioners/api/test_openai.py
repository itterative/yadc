import mock

from yadc.core import DatasetImage
from yadc.captioners.api import OpenAICaptioner

def test_koboldcpp(session, request_mocker, load_test_data):
    request_mocker.register_uri('GET', 'mock://koboldcpp/.well-known/serviceinfo', json={'software': {'name': 'koboldcpp'}})
    request_mocker.register_uri('GET', 'mock://koboldcpp/v1/models', json={'data': [{'id': 'koboldcpp/gemma-3-27b', 'object': 'model', 'owned_by': 'koboldcpp'}]})

    request_mocker.register_uri('POST', 'mock://koboldcpp/v1/chat/completions', text=load_test_data('koboldcpp.txt'))

    captioner = OpenAICaptioner(api_url='mock://koboldcpp/v1', session=session)
    captioner.load_model('koboldcpp/gemma-3-27b')

    expected = load_test_data('koboldcpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_koboldcpp_should_load_model(session, request_mocker, load_test_data):
    loaded_model = 'inactive'

    def load_model(model: str):
        def _load_model(r, c):
            nonlocal loaded_model
            loaded_model = model
            return {'success': True}
        
        return _load_model

    request_mocker.register_uri('GET', 'mock://koboldcpp/.well-known/serviceinfo', json={'software': {'name': 'koboldcpp'}})
    request_mocker.register_uri('GET', 'mock://koboldcpp/v1/models', json=lambda r, c: {'data': [{'id': loaded_model, 'object': 'model', 'owned_by': 'koboldcpp'}]})
    request_mocker.register_uri('GET', 'mock://koboldcpp/api/v1/model', json=lambda r, c: {'result': loaded_model})
    request_mocker.register_uri('GET', 'mock://koboldcpp/api/admin/list_options', json=['inactive', 'koboldcpp/gemma-3-27b'])

    request_mocker.register_uri('POST', 'mock://koboldcpp/api/admin/reload_config', json=load_model('koboldcpp/gemma-3-27b'))

    request_mocker.register_uri('POST', 'mock://koboldcpp/v1/chat/completions', text=load_test_data('koboldcpp.txt'))

    captioner = OpenAICaptioner(api_url='mock://koboldcpp/v1', session=session)
    captioner.load_model('koboldcpp/gemma-3-27b')

    expected = load_test_data('koboldcpp_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'

def test_openrouter_gpt_5_mini(session, request_mocker, load_test_data):
    request_mocker.register_uri('GET', 'mock://openrouter.ai/api/v1/credits', json={'data': {'total_credits': 1.0, 'total_usage': 0.1}})
    request_mocker.register_uri('GET', 'mock://openrouter.ai/api/v1/models', json={'data': [{'id': 'openai/gpt-5-mini', 'object': 'model', 'owned_by': 'koboldcpp'}]})

    request_mocker.register_uri('POST', 'mock://openrouter.ai/api/v1/chat/completions', text=load_test_data('openrouter_gpt_5_mini.txt'))

    captioner = OpenAICaptioner(api_url='mock://openrouter.ai/api/v1', session=session)
    captioner.load_model('openai/gpt-5-mini')

    expected = load_test_data('openrouter_gpt_5_mini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'
