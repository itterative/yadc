from typing import Callable

import pytest

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_llamacpp(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-llamacpp')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_llamacpp_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-llamacpp')
    cli_cmd('caption test_pedro.dataset --stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_koboldcpp(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-koboldcpp')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_koboldcpp_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-koboldcpp')
    cli_cmd('caption test_pedro.dataset --stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_vllm(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-vllm')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_vllm_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-vllm')
    cli_cmd('caption test_pedro.dataset --stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_ollama(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-ollama')
    cli_cmd('caption test_pedro.dataset --no-stream')

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_ollama_streaming(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-ollama')
    cli_cmd('caption test_pedro.dataset --stream')
