from typing import Callable

import pytest


class TestOfficialCLIBackends:
    """CLI integration tests for official API backends (Gemini, OpenRouter,
    OpenAI). Each backend is exercised in both streaming and non-streaming
    modes against the ``test_pedro`` dataset."""

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_gemini(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-gemini")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_gemini_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-gemini")
        cli_cmd("caption test_pedro.dataset --stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_openrouter(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-openrouter")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_openrouter_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-openrouter")
        cli_cmd("caption test_pedro.dataset --stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_openai(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-openai")
        cli_cmd("caption test_pedro.dataset --no-stream")

    @pytest.mark.timeout(60)
    @pytest.mark.integration_test
    def test_pedro_openai_streaming(self, cli):
        cli_cmd: Callable[[str], str] = cli(env="integration-tests-openai")
        cli_cmd("caption test_pedro.dataset --stream")
