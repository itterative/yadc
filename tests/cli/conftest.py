import pytest

@pytest.fixture(scope='function')
def cli():
    import pathlib
    import subprocess

    test_data_cwd = str(pathlib.Path(__file__).parent / 'test_data')

    current_env: str|None = None

    def _cmd(cmd: str, should_fail: bool = False):
        cmd = f'yadc {cmd}' if not current_env else f'yadc {cmd} --env {current_env}'

        try:
            data = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, encoding=None, errors=None, cwd=test_data_cwd)

            print('command succeeded:', cmd)
            print('command output:', data)

            assert not should_fail, f'command succeeded, but was expected to fail: {cmd}'
            return data
        except subprocess.CalledProcessError as ex:
            print('command failed:', cmd)
            print('command output:', ex.output)

            assert should_fail, f'command failed, but was expected to succeed: {cmd}'
            return ex.output


    def _cli(env: str):
        nonlocal current_env

        envs = _cmd('envs list').splitlines()
        if env not in envs:
            pytest.skip(f"env {env} not found; available: {', '.join(envs)}")

        current_env = env

        # ensure these exist
        _cmd('envs get api_url')
        _cmd('envs get api_token')
        _cmd('envs get api_model_name')

        return _cmd

    return _cli

@pytest.fixture(scope='session', autouse=True)
def cleanup_cli():
    yield

    import pathlib
    test_data = pathlib.Path(__file__).parent / 'test_data'

    for ext in ['.txt', '.toml', '.toml~', '.history~']:
        for file in test_data.glob(f'*{ext}'):
            file.unlink()
