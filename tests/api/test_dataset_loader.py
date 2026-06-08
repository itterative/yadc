"""Tests for ``DatasetLoader`` — read-only access to a dataset's TOML config.

The loader is pure: no DB, no filesystem writes, no side effects
beyond a logger warning on parse failure. These tests need only
``tmp_path`` and the standard ``logging_factory`` fixture.
"""

from pathlib import Path

import pytest

from yadc.api.services.dataset_loader import DatasetLoader


@pytest.fixture
def loader(logging_factory):
    return DatasetLoader(logging=logging_factory)


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


class TestLoadConfig:
    """``load_config`` — parse a dataset config TOML."""

    def test_parses_v2_config(self, loader, tmp_path):
        cfg_path = _write_config(tmp_path / "config.toml", '[[dataset]]\npath = "/data/img"\n')
        config = loader.load_config(cfg_path)
        assert config is not None
        assert len(config.dataset) == 1
        assert config.dataset[0].path == "/data/img"

    def test_returns_none_for_missing_file(self, loader, tmp_path):
        """A nonexistent config file returns ``None`` rather than raising."""
        assert loader.load_config(tmp_path / "missing.toml") is None

    def test_returns_none_for_malformed_toml(self, loader, tmp_path):
        """Garbage TOML is logged at WARN and returns ``None``.

        The caller treats this as "dataset is broken, skip" rather
        than crashing the surrounding workflow.
        """
        cfg_path = _write_config(tmp_path / "bad.toml", "this is not valid toml {][")
        assert loader.load_config(cfg_path) is None

    def test_uses_relaxed_validation(self, loader, tmp_path):
        """Missing API/model/template fields parse cleanly under ``strict=False``."""
        # No api_url, model, template — these are required by the
        # strict validator but the webui creates configs without
        # them and fills them in later.
        cfg_path = _write_config(tmp_path / "config.toml", '[[dataset]]\npath = "/data/img"\n')
        assert loader.load_config(cfg_path) is not None


class TestLoadRawConfig:
    """``load_raw_config`` — read the file as a raw dict (no v2 validation)."""

    def test_returns_dict(self, loader, tmp_path):
        cfg_path = _write_config(tmp_path / "config.toml", '[[dataset]]\npath = "/x"\n')
        raw = loader.load_raw_config(cfg_path)
        assert isinstance(raw, dict)
        assert raw["dataset"] == [{"path": "/x"}]


class TestResolveRelativePaths:
    """``resolve_relative_paths`` — rewrite relative paths to absolute in a raw dict."""

    def test_resolves_v2_relative_path(self, loader, tmp_path):
        base = tmp_path
        raw = {"dataset": [{"path": "images"}]}
        loader.resolve_relative_paths(raw, base)
        assert raw["dataset"][0]["path"] == str((base / "images").resolve())

    def test_leaves_absolute_paths_untouched(self, loader, tmp_path):
        abs_path = "/abs/path/to/images"
        raw = {"dataset": [{"path": abs_path}]}
        loader.resolve_relative_paths(raw, tmp_path)
        assert raw["dataset"][0]["path"] == abs_path

    def test_resolves_v1_paths_list(self, loader, tmp_path):
        base = tmp_path
        raw = {"dataset": {"paths": ["a", "b/c"]}}
        loader.resolve_relative_paths(raw, base)
        assert raw["dataset"]["paths"] == [
            str((base / "a").resolve()),
            str((base / "b/c").resolve()),
        ]

    def test_preserves_v1_absolute_paths(self, loader, tmp_path):
        raw = {"dataset": {"paths": ["/abs", "rel"]}}
        loader.resolve_relative_paths(raw, tmp_path)
        assert raw["dataset"]["paths"][0] == "/abs"
        assert raw["dataset"]["paths"][1] == str((tmp_path / "rel").resolve())


class TestGetDatasetPaths:
    """``get_dataset_paths`` — extract image directories from a parsed config."""

    def test_returns_declared_paths(self, loader, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        cfg_path = _write_config(tmp_path / "config.toml", f'[[dataset]]\npath = "{img_dir}"\n')
        config = loader.load_config(cfg_path)
        paths = loader.get_dataset_paths(config, cfg_path)
        assert str(img_dir.resolve()) in paths

    def test_returns_empty_for_none_config(self, loader):
        assert loader.get_dataset_paths(None) == []

    def test_drops_nonexistent_directories(self, loader, tmp_path):
        cfg_path = _write_config(tmp_path / "config.toml", '[[dataset]]\npath = "/does/not/exist"\n')
        config = loader.load_config(cfg_path)
        paths = loader.get_dataset_paths(config, cfg_path)
        assert paths == []

    def test_resolves_relative_paths_against_config_parent(self, loader, tmp_path):
        """A relative path in the config is resolved relative to the config's parent dir."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        cfg_path = _write_config(tmp_path / "config.toml", '[[dataset]]\npath = "images"\n')
        config = loader.load_config(cfg_path)
        paths = loader.get_dataset_paths(config, cfg_path)
        assert str(img_dir.resolve()) in paths
