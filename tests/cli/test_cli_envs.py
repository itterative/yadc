import functools

import pytest


@functools.lru_cache(maxsize=1)
def _keyring_available() -> bool:
    try:
        import keyrings.alt.file  # noqa: F401

        return True
    except ImportError:
        return False


class TestEnvsBasic:
    def test_envs_list_empty(self, cli):
        runner = cli(isolated=True)
        result = runner("envs list")
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_envs_set_and_get(self, cli):
        runner = cli(isolated=True)
        r1 = runner("envs set api_url http://localhost:5000")
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs get api_url")
        assert r2.returncode == 0, r2.stderr
        assert r2.stdout.strip() == "http://localhost:5000"

    def test_envs_show(self, cli):
        runner = cli(isolated=True)
        r1 = runner("envs set api_url http://localhost:5000")
        assert r1.returncode == 0

        r2 = runner("envs show")
        assert r2.returncode == 0, r2.stderr
        assert "api_url" in r2.stdout

    def test_envs_delete(self, cli):
        runner = cli(isolated=True)
        r1 = runner("envs set api_url http://localhost:5000")
        assert r1.returncode == 0

        r2 = runner("envs delete api_url")
        assert r2.returncode == 0, r2.stderr

        r3 = runner("envs get api_url", should_fail=True)
        assert r3.returncode != 0

    def test_envs_clear(self, cli):
        runner = cli(isolated=True)
        r1 = runner("envs set api_url http://localhost:5000")
        assert r1.returncode == 0

        r2 = runner("envs clear")
        assert r2.returncode == 0, r2.stderr

        r3 = runner("envs list")
        assert r3.returncode == 0
        assert r3.stdout.strip() == ""


class TestKeyMode:
    def test_key_mode_get_default(self, cli):
        runner = cli(isolated=True)
        result = runner("envs key-mode get")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "keyring"

    @pytest.mark.skipif(
        not _keyring_available(),
        reason="No keyring backend available (e.g. gpg/keyrings.alt missing)",
    )
    def test_key_mode_set_password(self, cli):
        runner = cli(isolated=True)
        r1 = runner(
            "envs key-mode set password",
            env_vars={"YADC_PASSWORD": "testpass123"},
        )
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs key-mode get")
        assert r2.returncode == 0, r2.stderr
        assert r2.stdout.strip() == "password"

    @pytest.mark.skipif(
        not _keyring_available(),
        reason="No keyring backend available (e.g. gpg/keyrings.alt missing)",
    )
    def test_key_mode_set_password_no_password_warns(self, cli):
        runner = cli(isolated=True)
        r1 = runner("envs key-mode set password")
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs key-mode get")
        assert r2.returncode == 0, r2.stderr
        assert r2.stdout.strip() == "password"

    @pytest.mark.skipif(
        not _keyring_available(),
        reason="No keyring backend available (e.g. gpg/keyrings.alt missing)",
    )
    def test_key_mode_roundtrip_keyring_to_password_and_back(self, cli):
        runner = cli(isolated=True)

        r1 = runner(
            "envs key-mode set password",
            env_vars={"YADC_PASSWORD": "mypassword"},
        )
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs set api_token secretvalue")
        assert r2.returncode == 0, r2.stderr

        r3 = runner(
            "envs key-mode set keyring",
            env_vars={"YADC_PASSWORD": "mypassword"},
        )
        assert r3.returncode == 0, r3.stderr

        r4 = runner("envs key-mode get")
        assert r4.returncode == 0, r4.stderr
        assert r4.stdout.strip() == "keyring"


class TestPasswordEncryption:
    @pytest.mark.skipif(
        not _keyring_available(),
        reason="No keyring backend available (e.g. gpg/keyrings.alt missing)",
    )
    def test_get_encrypted_token_with_password_env(self, cli):
        runner = cli(isolated=True)

        r1 = runner(
            "envs key-mode set password",
            env_vars={"YADC_PASSWORD": "sekrit"},
        )
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs set api_token mytoken")
        assert r2.returncode == 0, r2.stderr

        r3 = runner(
            "envs get api_token",
            env_vars={"YADC_PASSWORD": "sekrit"},
        )
        assert r3.returncode == 0, r3.stderr
        assert r3.stdout.strip() == "mytoken"

    @pytest.mark.skipif(
        not _keyring_available(),
        reason="No keyring backend available (e.g. gpg/keyrings.alt missing)",
    )
    def test_get_encrypted_token_without_password_prompts(self, cli):
        runner = cli(isolated=True)

        r1 = runner(
            "envs key-mode set password",
            env_vars={"YADC_PASSWORD": "sekrit"},
        )
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs set api_token mytoken")
        assert r2.returncode == 0, r2.stderr

        r3 = runner("envs get api_token", stdin="sekrit\n")
        assert r3.returncode == 0, r3.stderr
        assert r3.stdout.strip() == "mytoken"

    @pytest.mark.skipif(
        not _keyring_available(),
        reason="No keyring backend available (e.g. gpg/keyrings.alt missing)",
    )
    def test_get_encrypted_token_wrong_password_fails(self, cli):
        runner = cli(isolated=True)

        r1 = runner(
            "envs key-mode set password",
            env_vars={"YADC_PASSWORD": "correct"},
        )
        assert r1.returncode == 0, r1.stderr

        r2 = runner("envs set api_token mytoken")
        assert r2.returncode == 0, r2.stderr

        r3 = runner(
            "envs get api_token",
            env_vars={"YADC_PASSWORD": "wrong"},
            should_fail=True,
        )
        assert r3.returncode != 0
