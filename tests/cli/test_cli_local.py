from typing import Callable

import pytest


class TestLocalCLIBackends:
    """CLI integration tests for local model server backends (llama.cpp,
    KoboldCpp, vLLM, Ollama). Each backend is exercised in both streaming
    and non-streaming modes against the ``test_pedro`` dataset."""

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_llamacpp(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-llamacpp")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_llamacpp_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-llamacpp")
        cli_cmd("caption test_pedro.dataset --stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_koboldcpp(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-koboldcpp")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_koboldcpp_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-koboldcpp")
        cli_cmd("caption test_pedro.dataset --stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_vllm(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-vllm")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_vllm_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-vllm")
        cli_cmd("caption test_pedro.dataset --stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_ollama(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-ollama")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_ollama_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-local-ollama")
        cli_cmd("caption test_pedro.dataset --stream")
