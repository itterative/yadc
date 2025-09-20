import pytest
import requests
import requests_mock

@pytest.fixture
def load_test_data():
    def _load_test_data(case: str):
        import pathlib

        package_root = pathlib.Path(__file__).parent

        with open(package_root / 'test_data' / case, 'r') as f:
            return f.read()
    
    return _load_test_data

@pytest.fixture(scope='function')
def session():
    return requests.Session()

@pytest.fixture(scope='function')
def request_mocker(session):
    adapter = requests_mock.Adapter()
    session.mount('mock://', adapter)

    return adapter
