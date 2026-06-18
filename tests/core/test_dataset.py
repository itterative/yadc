"""Tests for DatasetImage — TOML serialization and history round-tripping.

``dump_toml`` serializes ``__pydantic_extra__`` to a TOML string.
``save_history`` / ``read_history`` persist extras + caption to the
``.history~`` sidecar. These tests verify that values survive a
dump → load round-trip for all supported types.
"""

from pathlib import Path

import tomlkit

from yadc.core.dataset import DatasetImage


class TestDumpTomlRoundTrip:
    """Create a DatasetImage with extra fields, dump to TOML, load back, verify."""

    def test_string(self):
        img = DatasetImage(path="/fake/img.jpg", artist="abc")
        parsed = tomlkit.loads(img.dump_toml())
        assert parsed["artist"] == "abc"

    def test_integer(self):
        img = DatasetImage(path="/fake/img.jpg", year=1872)
        parsed = tomlkit.loads(img.dump_toml())
        assert parsed["year"] == 1872

    def test_float(self):
        img = DatasetImage(path="/fake/img.jpg", score=3.14)
        parsed = tomlkit.loads(img.dump_toml())
        assert parsed["score"] == 3.14

    def test_boolean(self):
        img = DatasetImage(path="/fake/img.jpg", nsfw=True)
        parsed = tomlkit.loads(img.dump_toml())
        assert parsed["nsfw"] is True

    def test_list_of_strings(self):
        img = DatasetImage(path="/fake/img.jpg", tags=["foo", "bar", "baz"])
        parsed = tomlkit.loads(img.dump_toml())
        assert parsed["tags"] == ["foo", "bar", "baz"]

    def test_multiple_fields(self):
        img = DatasetImage(
            path="/fake/img.jpg",
            artist="Monet",
            style="impressionism",
            year=1872,
            tags=["painting", "landscape"],
        )
        parsed = tomlkit.loads(img.dump_toml())
        assert parsed["artist"] == "Monet"
        assert parsed["style"] == "impressionism"
        assert parsed["year"] == 1872
        assert parsed["tags"] == ["painting", "landscape"]

    def test_with_caption(self):
        img = DatasetImage(path="/fake/img.jpg", artist="abc")
        img.caption = "a painting"
        parsed = tomlkit.loads(img.dump_toml(with_caption=True))
        assert parsed["artist"] == "abc"
        assert parsed["caption"] == "a painting"

    def test_without_caption(self):
        img = DatasetImage(path="/fake/img.jpg", artist="abc")
        img.caption = "should not appear"
        parsed = tomlkit.loads(img.dump_toml(with_caption=False))
        assert "caption" not in parsed

    def test_with_caption_does_not_mutate_extras(self):
        # dump_toml is a serializer and must not write back into
        # __pydantic_extra__. Previously the ``caption`` key (and any mutable
        # tomlkit values) leaked in, so a later save_history() + update_caption()
        # wrote a stale caption into the live TOML sidecar.
        img = DatasetImage(path="/fake/img.jpg", artist="abc", tags=["x", "y"])
        img.caption = "a painting"

        img.dump_toml(with_caption=True)

        extras = img.__pydantic_extra__ or {}
        assert "caption" not in extras
        assert extras["artist"] == "abc"
        assert extras["tags"] == ["x", "y"]


class TestSaveHistoryRoundTrip:
    """save_history → read_history must preserve extras and caption."""

    @staticmethod
    def _touch(tmp_path: Path, name: str = "img.jpg") -> Path:
        p = tmp_path / name
        p.touch()
        return p

    def test_preserves_string_extras(self, tmp_path: Path):
        img_path = self._touch(tmp_path)
        img = DatasetImage(path=str(img_path), artist="Monet")
        img.caption = "a painting"
        img.save_history()

        history = img.read_history()
        assert len(history) == 1
        assert history[0].caption == "a painting"
        assert (history[0].__pydantic_extra__ or {})["artist"] == "Monet"

    def test_preserves_multiple_types(self, tmp_path: Path):
        img_path = self._touch(tmp_path)
        img = DatasetImage(
            path=str(img_path),
            artist="Monet",
            style="impressionism",
            year=1872,
            tags=["painting", "landscape"],
        )
        img.caption = "a painting"
        img.save_history()

        extras = img.read_history()[0].__pydantic_extra__ or {}
        assert extras["artist"] == "Monet"
        assert extras["style"] == "impressionism"
        assert extras["year"] == 1872
        assert extras["tags"] == ["painting", "landscape"]

    def test_multiple_entries(self, tmp_path: Path):
        img_path = self._touch(tmp_path)

        DatasetImage(path=str(img_path), artist="abc").save_history()
        DatasetImage(path=str(img_path), artist="xyz").save_history()

        history = DatasetImage(path=str(img_path)).read_history()
        assert len(history) == 2
        assert (history[0].__pydantic_extra__ or {})["artist"] == "abc"
        assert (history[1].__pydantic_extra__ or {})["artist"] == "xyz"
