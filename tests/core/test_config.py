import pathlib

import pydantic
import pytest
import toml

from yadc.core.config import Config, ConfigDatasetEntry, parse_config

TEST_DATA = pathlib.Path(__file__).parent / 'test_data'


def _load(name: str) -> dict:
    with open(TEST_DATA / name) as f:
        return toml.loads(f.read())


class TestParseConfigV2:
    def test_minimal(self):
        raw = _load('v2_minimal.toml')
        cfg = parse_config(raw)

        assert isinstance(cfg, Config)
        assert len(cfg.dataset) == 1
        assert cfg.dataset[0].path == '/tmp/dataset_a'
        assert cfg.dataset[0].images == []
        assert cfg.dataset[0].extras == {}

    def test_full(self):
        raw = _load('v2_full.toml')
        cfg = parse_config(raw)

        assert cfg.interactive is True
        assert cfg.rounds == 2
        assert cfg.caption_suffix == '.caption'
        assert cfg.overwrite_captions is True
        assert cfg.settings.max_tokens == 1024

        assert len(cfg.dataset) == 3

        # first entry: path + extras
        entry = cfg.dataset[0]
        assert entry.path == '/tmp/dataset_chars'
        assert entry.extras == {'style': 'anime', 'universe': 'genshin'}
        assert entry.images == []

        # second entry: path + different extras
        entry = cfg.dataset[1]
        assert entry.path == '/tmp/dataset_scenery'
        assert entry.extras == {'style': 'photo'}

        # third entry: inline images, no path
        entry = cfg.dataset[2]
        assert entry.path == ''
        assert len(entry.images) == 2
        assert entry.images[0].path == 'image_1.png'
        assert entry.images[0].__pydantic_extra__ == {'name': 'alice'}
        assert entry.images[1].path == 'image_2.png'
        assert entry.images[1].__pydantic_extra__ == {'name': 'bob', 'style': 'sketch'}

    def test_no_dataset(self):
        raw = _load('v2_no_dataset.toml')
        cfg = parse_config(raw)

        assert cfg.dataset == []


class TestParseConfigV1:
    def test_basic(self):
        raw = _load('v1_basic.toml')
        cfg = parse_config(raw)

        assert isinstance(cfg, Config)

        # v1 paths become separate entries
        assert len(cfg.dataset) == 3
        assert cfg.dataset[0].path == '/tmp/dataset_1'
        assert cfg.dataset[1].path == '/tmp/dataset_2'

        # v1 inline images become a separate entry
        images_entry = cfg.dataset[2]
        assert images_entry.path == ''
        assert len(images_entry.images) == 1
        assert images_entry.images[0].path == 'inline_image.png'
        assert images_entry.images[0].__pydantic_extra__ == {'foo': 'bar'}

    def test_empty(self):
        raw = _load('v1_empty.toml')
        cfg = parse_config(raw)

        assert isinstance(cfg, Config)
        assert cfg.dataset == []


class TestParseConfigInvalid:
    def test_invalid_raises(self):
        raw = _load('invalid.toml')

        with pytest.raises(pydantic.ValidationError):
            parse_config(raw)


class TestExtrasOverride:
    def test_dataset_extras_as_defaults(self):
        raw = _load('v2_extras_override.toml')
        cfg = parse_config(raw)

        entry = cfg.dataset[0]
        assert entry.extras == {'style': 'watercolor', 'artist': 'unknown'}
        assert len(entry.images) == 1

        # per-image 'artist' is set, 'style' is not — will be applied at load time
        img = entry.images[0]
        assert img.__pydantic_extra__ == {'artist': 'picasso'}
