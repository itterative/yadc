"""Tests for tag-result postprocessing (underscore replacement)."""

from yadc.taggers.base import TaggerResult
from yadc.taggers.postprocessing import replace_underscores


class TestReplaceUnderscores:
    def test_replace_underscores_converts_underscores_to_spaces(self) -> None:
        result = TaggerResult(
            tags={"1girl": 0.99, "long_hair": 0.9, "solo": 0.8},
            categories={"general": ["1girl", "long_hair", "solo"]},
        )
        out = replace_underscores(result)
        assert out.tags == {"1girl": 0.99, "long hair": 0.9, "solo": 0.8}
        assert out.categories == {"general": ["1girl", "long hair", "solo"]}

    def test_replace_underscores_preserves_kaomojis(self) -> None:
        result = TaggerResult(
            tags={"^_^": 0.7, ">_<": 0.6, "smile": 0.5},
            categories={"general": ["^_^", ">_<", "smile"]},
        )
        out = replace_underscores(result)
        # Kaomoji tags are untouched; non-kaomoji tags still convert (none here).
        assert out.tags == {"^_^": 0.7, ">_<": 0.6, "smile": 0.5}

    def test_replace_underscores_noop_returns_same_object(self) -> None:
        """When nothing would change, the original result is returned unchanged."""
        result = TaggerResult(tags={"1girl": 0.99, "solo": 0.8}, categories={"general": ["1girl", "solo"]})
        assert replace_underscores(result) is result

    def test_replace_underscores_preserves_scores(self) -> None:
        result = TaggerResult(tags={"long_hair": 0.42}, categories={})
        out = replace_underscores(result)
        assert out.tags["long hair"] == 0.42

    def test_replace_underscores_preserves_category_order(self) -> None:
        result = TaggerResult(
            tags={"long_hair": 0.9, "rei_(ayanami)": 0.8},
            categories={"general": ["long_hair"], "character": ["rei_(ayanami)"]},
        )
        out = replace_underscores(result)
        assert list(out.categories.keys()) == ["general", "character"]
        # Parens aren't underscores — only the underscore in the qualifier moves.
        assert out.categories["character"] == ["rei (ayanami)"]
