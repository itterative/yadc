from typing import Callable

import pytest

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_gemini(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-gemini')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_gemini_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-gemini')
    cli_cmd('caption test_pedro.dataset --stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_openrouter(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-openrouter')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_openrouter_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-openrouter')
    cli_cmd('caption test_pedro.dataset --stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_openai(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-openai')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_openai_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-openai')
    cli_cmd('caption test_pedro.dataset --stream')
