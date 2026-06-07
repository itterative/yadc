"""Tests for yadc.core.captioning.loader — config loading + override application + template resolution.

The loader is the shared foundation both the CLI and the API build on for
loading a dataset config, applying overrides, parsing the result,
resolving the dataset, and filtering the images to caption. These tests
focus on the three public functions in isolation — no DI, no event loop,
no HTTP.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import tomlkit

from yadc.cmd.envs.user_config import UserConfig, UserConfigApi
from yadc.core.captioning.loader import (
    apply_config_overrides,
    load_dataset_config,
    resolve_template,
)
from yadc.core.captioning.options import CaptionJobOptions

# Centralize patch target paths so renames only need to be updated here.
# The loader imports `cmd_envs` / `cmd_templates` / `cmd_configs` as module
# names, so the patches target the loader's namespace, not the originals.
_PATCH_CMD_ENVS = "yadc.core.captioning.loader.cmd_envs"
_PATCH_CMD_TEMPLATES = "yadc.core.captioning.loader.cmd_templates"
_PATCH_CMD_CONFIGS = "yadc.core.captioning.loader.cmd_configs"


def _user_config(url: str = "", token: str = "", model_name: str = "", max_concurrent: int | None = None) -> UserConfig:
    return UserConfig(
        api=UserConfigApi(
            url=url, token=token, model_name=model_name, max_concurrent=max_concurrent
        )
    )


def _real_image(path: Path) -> Path:
    """Create a minimal valid JPEG at *path* so resolve_dataset accepts it."""
    from PIL import Image

    Image.new("RGB", (1, 1), color="red").save(path, format="JPEG")
    return path


def _dataset_toml(scan_dir: Path, **dataset_extras) -> dict:
    """Build a minimal dataset config dict that scans images from *scan_dir*.

    Uses a single ``[[dataset]]`` entry with ``path = scan_dir`` rather
    than inline ``[[dataset.images]]`` blocks, so the loader's
    ``resolve_dataset`` discovers the images via the directory scan and
    sets their ``path`` to the absolute path under *scan_dir*. This
    matches how the API and CLI typically configure datasets.
    """
    return {
        "api": {"url": "http://example.com", "model_name": "m"},
        "prompt": {"template": "t"},
        "dataset": [
            {
                "path": str(scan_dir),
                **({"extras": dict(dataset_extras)} if dataset_extras else {}),
            },
        ],
    }


def _write_config(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(tomlkit.dumps(content))
    return path


class TestApplyConfigOverrides:
    """apply_config_overrides — merge env + CaptionJobOptions into raw TOML."""

    def test_opts_override_toml_when_env_empty(self):
        raw = {"api": {"url": "http://from-toml", "model_name": "m"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            apply_config_overrides(raw, CaptionJobOptions(api_url="http://from-opts"))
        assert raw["api"]["url"] == "http://from-opts"

    def test_env_overrides_toml(self):
        raw = {"api": {"url": "http://from-toml", "model_name": "m"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config(url="http://from-env")
            apply_config_overrides(raw, CaptionJobOptions())
        assert raw["api"]["url"] == "http://from-env"

    def test_opts_take_priority_over_env(self):
        raw = {"api": {"url": "http://from-toml", "model_name": "m"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config(url="http://from-env", token="tok")
            apply_config_overrides(raw, CaptionJobOptions(api_url="http://from-opts"))
        assert raw["api"]["url"] == "http://from-opts"
        # Token has no opt override — falls through to env.
        assert raw["api"]["token"] == "tok"

    def test_prompt_name_clears_template(self):
        raw = {"api": {"url": "x", "model_name": "m"}, "prompt": {"template": "old"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            apply_config_overrides(raw, CaptionJobOptions(prompt_name="newname"))
        assert raw["prompt"]["name"] == "newname"
        assert "template" not in raw["prompt"]

    def test_prompt_template_clears_name(self):
        raw = {"api": {"url": "x", "model_name": "m"}, "prompt": {"name": "old"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            apply_config_overrides(raw, CaptionJobOptions(prompt_template="new t"))
        assert raw["prompt"]["template"] == "new t"
        assert "name" not in raw["prompt"]

    def test_max_tokens_only_set_when_changed(self):
        raw = {"api": {"url": "x", "model_name": "m"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            apply_config_overrides(raw, CaptionJobOptions())  # default 512
            assert "settings" not in raw
            apply_config_overrides(raw, CaptionJobOptions(max_tokens=1024))
            assert raw["settings"]["max_tokens"] == 1024

    def test_rounds_only_set_when_changed(self):
        raw = {"api": {"url": "x", "model_name": "m"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            apply_config_overrides(raw, CaptionJobOptions(rounds=1))  # default
            assert "rounds" not in raw
            apply_config_overrides(raw, CaptionJobOptions(rounds=3))
            assert raw["rounds"] == 3

    def test_reasoning_set_only_when_enabled(self):
        raw = {"api": {"url": "x", "model_name": "m"}}
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            apply_config_overrides(raw, CaptionJobOptions())  # default False
            assert "reasoning" not in raw
            apply_config_overrides(raw, CaptionJobOptions(reasoning=True, reasoning_effort="high"))
            assert raw["reasoning"]["enable"] is True
            assert raw["reasoning"]["thinking_effort"] == "high"


class TestMaxConcurrentResolution:
    """``apply_config_overrides`` resolves ``opts.max_concurrent`` from
    ``None`` (sentinel) to a concrete int using the env as a fallback.

    Contract: explicit opts value > env value > 1 (sequential). The
    resolution happens in-place on ``opts`` so the runner always
    receives a valid ``int >= 1`` regardless of what the caller set.
    """

    def test_none_resolves_to_env_value(self):
        """``opts.max_concurrent=None`` + env has 4 → opts becomes 4."""
        opts = CaptionJobOptions()  # default None
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config(max_concurrent=4)
            apply_config_overrides({"api": {"url": "x", "model_name": "m"}}, opts)
        assert opts.max_concurrent == 4

    def test_none_resolves_to_one_when_env_unset(self):
        """``opts.max_concurrent=None`` + env has no value → opts becomes 1."""
        opts = CaptionJobOptions()  # default None
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()  # max_concurrent=None
            apply_config_overrides({"api": {"url": "x", "model_name": "m"}}, opts)
        assert opts.max_concurrent == 1

    def test_explicit_opts_value_wins_over_env(self):
        """Caller-set ``max_concurrent=8`` overrides the env's 4."""
        opts = CaptionJobOptions(max_concurrent=8)
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config(max_concurrent=4)
            apply_config_overrides({"api": {"url": "x", "model_name": "m"}}, opts)
        assert opts.max_concurrent == 8

    def test_explicit_one_keeps_one(self):
        """Caller-set ``max_concurrent=1`` stays 1 even if env has a different value.

        This is the one edge of the contract where the env cannot
        override the caller — the caller explicitly opted into
        sequential. Distinguishable from "not set" because
        ``CaptionJobOptions`` defaults to ``None``.
        """
        opts = CaptionJobOptions(max_concurrent=1)
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config(max_concurrent=4)
            apply_config_overrides({"api": {"url": "x", "model_name": "m"}}, opts)
        assert opts.max_concurrent == 1

    def test_invalid_env_value_falls_back_to_one(self):
        """Env's ``max_concurrent=0`` (or negative) is treated as unset.

        The loader doesn't try to validate or coerce — it just falls
        back to 1 to avoid passing an invalid value to the runner
        (which would raise ``ValueError``). The PUT endpoint rejects
        invalid values at the boundary with ``Field(ge=1)``.
        """
        for bad in (0, -1, -100):
            opts = CaptionJobOptions()
            with patch(_PATCH_CMD_ENVS) as mock_env:
                mock_env.load_env.return_value = _user_config(max_concurrent=bad)
                apply_config_overrides({"api": {"url": "x", "model_name": "m"}}, opts)
            assert opts.max_concurrent == 1, f"env value {bad} should fall back to 1"


class TestResolveTemplate:
    """resolve_template — fallback chain (template → user → builtin → default)."""

    def test_returns_explicit_template(self):
        assert resolve_template("name", "explicit") == "explicit"

    def test_uses_user_template(self):
        with patch(_PATCH_CMD_TEMPLATES) as mock_t:
            mock_t.load_user_template.return_value = "user t"
            assert resolve_template("name", "") == "user t"

    def test_falls_back_to_builtin(self):
        with patch(_PATCH_CMD_TEMPLATES) as mock_t:
            mock_t.load_user_template.side_effect = Exception("not found")
            mock_t.load_builtin_template.return_value = "builtin t"
            assert resolve_template("name", "") == "builtin t"

    def test_raises_on_named_template_not_found(self):
        with patch(_PATCH_CMD_TEMPLATES) as mock_t:
            mock_t.load_user_template.side_effect = Exception("not found")
            mock_t.load_builtin_template.side_effect = Exception("not found")
            mock_t.list_user_template.return_value = ["other1", "other2"]
            with pytest.raises(ValueError, match="Prompt template 'missing' not found"):
                resolve_template("missing", "")

    def test_uses_default_when_no_name(self):
        with patch(_PATCH_CMD_TEMPLATES) as mock_t:
            mock_t.load_user_template.side_effect = Exception("not found")
            mock_t.load_builtin_template.side_effect = Exception("not found")
            mock_t.default_template.return_value = "default t"
            assert resolve_template("", "") == "default t"


class TestLoadDatasetConfig:
    """load_dataset_config — end-to-end: load + apply + parse + resolve + filter."""

    def test_happy_path(self, tmp_path):
        img1 = _real_image(tmp_path / "img1.jpg")
        img2 = _real_image(tmp_path / "img2.jpg")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))

        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            cfg, images = load_dataset_config(config, CaptionJobOptions())

        assert cfg.api.url == "http://example.com"
        assert {img.path for img in images} == {str(img1), str(img2)}

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset_config(tmp_path / "missing.toml", CaptionJobOptions())

    def test_invalid_config_raises_value_error(self, tmp_path):
        # Missing required api.url / api.model_name triggers ValidationError.
        config = _write_config(tmp_path, {"prompt": {"template": "t"}})
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            with pytest.raises(ValueError, match="invalid configuration"):
                load_dataset_config(config, CaptionJobOptions())

    def test_user_config_merges(self, tmp_path):
        img = _real_image(tmp_path / "img.jpg")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))

        merged_raw = {
            "api": {"url": "http://from-user-cfg", "model_name": "m"},
            "prompt": {"template": "t"},
            "dataset": [{"path": str(tmp_path)}],
        }
        with patch(_PATCH_CMD_CONFIGS) as mock_cfgs:
            mock_cfgs.merge_user_config.return_value = merged_raw
            with patch(_PATCH_CMD_ENVS) as mock_env:
                mock_env.load_env.return_value = _user_config()
                cfg, images = load_dataset_config(config, CaptionJobOptions(), user_config="myuser")
        mock_cfgs.merge_user_config.assert_called_once()
        assert cfg.api.url == "http://from-user-cfg"
        assert len(images) == 1
        assert images[0].path == str(img)

    def test_user_config_merge_failure_reraises(self, tmp_path):
        _real_image(tmp_path / "img.jpg")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))
        with patch(_PATCH_CMD_CONFIGS) as mock_cfgs:
            mock_cfgs.merge_user_config.side_effect = ValueError("user config not found")
            with patch(_PATCH_CMD_ENVS) as mock_env:
                mock_env.load_env.return_value = _user_config()
                with pytest.raises(ValueError, match="user config not found"):
                    load_dataset_config(config, CaptionJobOptions(), user_config="missing")

    def test_image_ids_filters_to_resolved_paths(self, tmp_path):
        img1 = _real_image(tmp_path / "img1.jpg")
        img2 = _real_image(tmp_path / "img2.jpg")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))

        def resolver(image_id: int):
            return {1: str(img1), 2: str(img2)}.get(image_id)

        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            _, images = load_dataset_config(
                config,
                CaptionJobOptions(image_ids=[2]),
                image_path_resolver=resolver,
            )
        assert len(images) == 1
        assert images[0].path == str(img2)

    def test_image_ids_empty_resolver_raises(self, tmp_path):
        _real_image(tmp_path / "img1.jpg")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))
        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            with pytest.raises(ValueError, match="Specified image"):
                load_dataset_config(
                    config,
                    CaptionJobOptions(image_ids=[99]),
                    image_path_resolver=lambda _id: None,
                )

    def test_overwrite_false_skips_captioned(self, tmp_path):
        _real_image(tmp_path / "img1.jpg")
        img2 = _real_image(tmp_path / "img2.jpg")
        (tmp_path / "img1.txt").write_text("existing caption")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))

        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            _, images = load_dataset_config(config, CaptionJobOptions(overwrite=False))
        assert len(images) == 1
        assert images[0].path == str(img2)

    def test_overwrite_true_includes_captioned(self, tmp_path):
        img1 = _real_image(tmp_path / "img1.jpg")
        (tmp_path / "img1.txt").write_text("existing caption")
        config = _write_config(tmp_path, _dataset_toml(tmp_path))

        with patch(_PATCH_CMD_ENVS) as mock_env:
            mock_env.load_env.return_value = _user_config()
            _, images = load_dataset_config(config, CaptionJobOptions(overwrite=True))
        assert len(images) == 1
        assert images[0].path == str(img1)
