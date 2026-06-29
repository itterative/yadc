"""Tests for tag draft formatters and extras-shape helpers."""

from __future__ import annotations

import textwrap

import pytest

from yadc.taggers.base import TaggerResult
from yadc.taggers.formatters import (
    available_draft_formats,
    comma_formatter,
    extras_tags,
    format_draft,
    get_draft_formatter,
    scored_formatter,
    structured_formatter,
    top_rating,
)


def _result() -> TaggerResult:
    return TaggerResult(
        tags={
            "general": 0.99,
            "sensitive": 0.6,
            "questionable": 0.3,
            "explicit": 0.1,
            "1girl": 0.95,
            "solo": 0.8,
            "hatsune_miku": 0.9,
        },
        categories={
            "rating": ["general", "sensitive", "questionable", "explicit"],
            "general": ["1girl", "solo"],
            "character": ["hatsune_miku"],
        },
    )


class TestRegistry:
    def test_all_three_registered(self) -> None:
        assert {"comma", "structured", "scored"} <= set(available_draft_formats())

    def test_get_known_formatter(self) -> None:
        assert get_draft_formatter("comma") is comma_formatter
        assert get_draft_formatter("scored") is scored_formatter

    def test_get_unknown_formatter_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown draft format"):
            get_draft_formatter("nope")

    def test_format_draft_dispatches_by_name(self) -> None:
        assert format_draft("comma", _result()) == comma_formatter(_result())
        assert format_draft("scored", _result()) == scored_formatter(_result())


class TestCommaFormatter:
    def test_excludes_rating_character_before_general(self) -> None:
        # Rating dropped; character leads, then general (canonical order).
        assert comma_formatter(_result()) == "hatsune_miku, 1girl, solo"

    def test_empty_result(self) -> None:
        assert comma_formatter(TaggerResult(tags={}, categories={})) == ""

    def test_falls_back_to_tags_order_without_categories(self) -> None:
        result = TaggerResult(tags={"a": 0.9, "b": 0.5}, categories={})
        assert comma_formatter(result) == "a, b"

    def test_sorts_alphabetically_within_each_section(self) -> None:
        """Tags appear in alphabetical order within each section (label-file order doesn't bleed through)."""
        result = TaggerResult(
            tags={"solo": 0.9, "1girl": 0.95, "long_hair": 0.8, "smile": 0.7, "rei": 0.85, "asuka": 0.86},
            categories={
                "general": ["solo", "1girl", "long_hair", "smile"],  # not alphabetical
                "character": ["rei", "asuka"],  # not alphabetical
            },
        )
        # Each section sorted alphabetically; character leads general.
        assert comma_formatter(result) == "asuka, rei, 1girl, long_hair, smile, solo"

    def test_uncategorized_result_is_sorted_alphabetically(self) -> None:
        """Uncategorized model output (no categories dict) is sorted alphabetically."""
        result = TaggerResult(tags={"zebra": 0.5, "apple": 0.9, "mango": 0.7}, categories={})
        assert comma_formatter(result) == "apple, mango, zebra"


class TestStructuredFormatter:
    def test_full_output_character_before_general(self) -> None:
        # Rating, then character, then general (canonical order).
        assert structured_formatter(_result()) == textwrap.dedent("""\
            rating: general
            character: hatsune_miku
            general: 1girl, solo""")

    def test_omits_empty_sections(self) -> None:
        result = TaggerResult(
            tags={"1girl": 0.9},
            categories={"general": ["1girl"]},
        )
        assert structured_formatter(result) == "general: 1girl"

    def test_sorts_alphabetically_within_each_section(self) -> None:
        """Tags within each category are sorted alphabetically, regardless of label-file order."""
        result = TaggerResult(
            tags={"solo": 0.9, "1girl": 0.95, "long_hair": 0.8, "rei": 0.85, "asuka": 0.86},
            categories={
                "general": ["solo", "1girl", "long_hair"],
                "character": ["rei", "asuka"],
            },
        )
        assert structured_formatter(result) == textwrap.dedent("""\
            character: asuka, rei
            general: 1girl, long_hair, solo""")


class TestScoredFormatter:
    def test_full_output_with_scores_character_before_general(self) -> None:
        # Like structured, but each tag carries a 2-decimal confidence.
        assert scored_formatter(_result()) == textwrap.dedent("""\
            rating: general (0.99)
            character: hatsune_miku (0.90)
            general: 1girl (0.95), solo (0.80)""")

    def test_omits_empty_sections(self) -> None:
        result = TaggerResult(
            tags={"1girl": 0.9},
            categories={"general": ["1girl"]},
        )
        assert scored_formatter(result) == "general: 1girl (0.90)"

    def test_omits_rating_when_none_survived(self) -> None:
        result = TaggerResult(
            tags={"1girl": 0.9},
            categories={"general": ["1girl"]},
        )
        assert "rating" not in scored_formatter(result)

    def test_empty_result(self) -> None:
        assert scored_formatter(TaggerResult(tags={}, categories={})) == ""

    def test_sorts_alphabetically_within_each_section(self) -> None:
        """Tags within each category are sorted alphabetically, with their scores."""
        result = TaggerResult(
            tags={"solo": 0.9, "1girl": 0.95, "long_hair": 0.8, "rei": 0.85, "asuka": 0.86},
            categories={
                "general": ["solo", "1girl", "long_hair"],
                "character": ["rei", "asuka"],
            },
        )
        assert scored_formatter(result) == textwrap.dedent("""\
            character: asuka (0.86), rei (0.85)
            general: 1girl (0.95), long_hair (0.80), solo (0.90)""")


class TestTopRating:
    def test_returns_highest_scored(self) -> None:
        assert top_rating(_result()) == "general"

    def test_none_when_no_rating_category(self) -> None:
        result = TaggerResult(tags={"1girl": 0.9}, categories={"general": ["1girl"]})
        assert top_rating(result) is None

    def test_none_when_rating_empty(self) -> None:
        result = TaggerResult(tags={"1girl": 0.9}, categories={"rating": [], "general": ["1girl"]})
        assert top_rating(result) is None


class TestExtrasTags:
    def test_shape_general_character_rating_string(self) -> None:
        assert extras_tags(_result()) == {
            "general": ["1girl", "solo"],
            "character": ["hatsune_miku"],
            "rating": "general",
        }

    def test_omits_rating_when_absent(self) -> None:
        result = TaggerResult(tags={"1girl": 0.9}, categories={"general": ["1girl"]})
        assert extras_tags(result) == {"general": ["1girl"], "character": []}

    def test_categories_sorted_alphabetically(self) -> None:
        """Category lists are sorted alphabetically so extras are diff-stable across re-runs."""
        result = TaggerResult(
            tags={"solo": 0.9, "1girl": 0.95, "long_hair": 0.8, "rei": 0.85, "asuka": 0.86},
            categories={
                "general": ["solo", "1girl", "long_hair"],
                "character": ["rei", "asuka"],
            },
        )
        assert extras_tags(result) == {
            "general": ["1girl", "long_hair", "solo"],
            "character": ["asuka", "rei"],
        }
