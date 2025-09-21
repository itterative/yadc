from typing import Callable

import pytest

@pytest.mark.timeout(60)
@pytest.mark.integration_test
def test_pedro_koboldcpp(cli):
    cli_cmd: Callable[[str], str] = cli(env='integration-tests-local-koboldcpp')
    cli_cmd('caption test_pedro.dataset')
