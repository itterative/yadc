import mock

from yadc.core import DatasetImage
from yadc.captioners.api import GeminiCaptioner

def test_gemini(session, request_mocker, load_test_data):
    request_mocker.register_uri('GET', 'mock://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash', json={'name': 'gemini-2.5-flash', 'version': '1', 'displayName': 'Gemini 2.5 Flash', 'supportedGenerationMethods': ['generateContent'], 'thinking': True})

    request_mocker.register_uri('POST', 'mock://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse', text=load_test_data('gemini.txt'))

    captioner = GeminiCaptioner(
        api_url='mock://generativelanguage.googleapis.com/v1beta',
        api_token='secret token',
        session=session,
    )

    captioner.load_model('gemini-2.5-flash')

    expected = load_test_data('gemini_result.txt')
    got = captioner.predict(mock.MagicMock(spec=DatasetImage))

    assert got == expected, 'bad prediction'
