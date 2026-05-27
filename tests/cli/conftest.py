import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest


class CLIResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def _run_yadc(
    cmd: str,
    *,
    cwd: str,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> CLIResult:
    full_cmd = ["uv", "run", "yadc", *cmd.split()]
    result = subprocess.run(
        full_cmd,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )
    return CLIResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


@pytest.fixture(scope="function")
def cli():
    """Fixture for running yadc CLI commands.

    Returns a callable with two modes:

    1. Integration mode (existing)::

           runner = cli(env="my-env")
           output = runner("caption dataset.toml")

       This runs against the *real* user config and is used by the
       network integration tests.

    2. Isolated mode (new)::

           runner = cli(isolated=True)
           result = runner("envs list")
           assert result.returncode == 0

       This runs in a temporary ``XDG_CONFIG_HOME`` / ``XDG_STATE_HOME``
       so the real user config is never touched.  The returned object is
       a ``CLIResult`` with ``returncode``, ``stdout``, and ``stderr``.
    """

    test_data_cwd = str(Path(__file__).parent / "test_data")

    # --- integration mode (backward compat) ---

    current_env: str | None = None

    def _integration_cmd(cmd: str, should_fail: bool = False):
        full_cmd = f"yadc {cmd}" if current_env is None else f"yadc {cmd} --env {current_env}"

        try:
            data = subprocess.check_output(
                full_cmd,
                shell=True,
                text=True,
                stderr=subprocess.STDOUT,
                encoding=None,
                errors=None,
                cwd=test_data_cwd,
            )
            print("command succeeded:", full_cmd)
            print("command output:", data)
            assert not should_fail, f"command succeeded, but was expected to fail: {full_cmd}"
            return data
        except subprocess.CalledProcessError as ex:
            print("command failed:", full_cmd)
            print("command output:", ex.output)
            assert should_fail, f"command failed, but was expected to succeed: {full_cmd}"
            return ex.output

    def _integration_cli(env: str):
        nonlocal current_env

        envs = _integration_cmd("envs list").splitlines()
        if env not in envs:
            pytest.skip(f"env {env} not found; available: {', '.join(envs)}")

        current_env = env
        _integration_cmd("envs get api_url")
        _integration_cmd("envs get api_token")
        _integration_cmd("envs get api_model_name")
        return _integration_cmd

    # --- isolated mode ---

    def _isolated_cli(
        *,
        initial_config: str | None = None,
    ) -> Callable[..., CLIResult]:
        tmpdir = tempfile.mkdtemp()
        config_home = Path(tmpdir) / "config"
        state_home = Path(tmpdir) / "state"
        data_home = Path(tmpdir) / "data"
        config_home.mkdir()
        state_home.mkdir()
        data_home.mkdir()

        yadc_config_dir = config_home / "yadc"
        yadc_config_dir.mkdir()
        config_file = yadc_config_dir / "config.toml"

        if initial_config is not None:
            config_file.write_text(initial_config)

        base_env = os.environ.copy()
        base_env["XDG_CONFIG_HOME"] = str(config_home)
        base_env["XDG_STATE_HOME"] = str(state_home)
        base_env["XDG_DATA_HOME"] = str(data_home)
        base_env["PYTHON_KEYRING_BACKEND"] = "keyrings.alt.file.PlaintextKeyring"

        def _isolated_cmd(
            cmd: str,
            *,
            stdin: str | None = None,
            env_vars: dict[str, str] | None = None,
            should_fail: bool = False,
        ) -> CLIResult:
            merged_env = base_env.copy()
            if env_vars:
                merged_env.update(env_vars)

            result = _run_yadc(cmd, cwd=test_data_cwd, env=merged_env, stdin=stdin)

            if should_fail:
                assert result.returncode != 0, (
                    f"command succeeded but was expected to fail: yadc {cmd}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
            else:
                assert result.returncode == 0, (
                    f"command failed but was expected to succeed: yadc {cmd}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
            return result

        return _isolated_cmd

    # --- public dispatcher ---

    def _cli(
        *,
        env: str | None = None,
        isolated: bool = False,
        initial_config: str | None = None,
    ) -> Callable[..., str] | Callable[..., CLIResult]:
        if isolated:
            return _isolated_cli(initial_config=initial_config)
        if env is not None:
            return _integration_cli(env)
        raise ValueError("Specify either env or isolated=True")

    return _cli


@pytest.fixture(scope="session", autouse=True)
def cleanup_cli():
    yield

    test_data = Path(__file__).parent / "test_data"

    for ext in [".txt", ".toml", ".toml~", ".history~"]:
        for file in test_data.glob(f"*{ext}"):
            file.unlink()
