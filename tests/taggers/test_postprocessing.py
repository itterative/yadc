"""Tests for tag-result postprocessing (underscore replacement + tag policy)."""

from yadc.taggers.base import TaggerResult
from yadc.taggers.postprocessing import TagPolicy, apply_policy, replace_underscores


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


class TestApplyPolicy:
    """``apply_policy`` — auto-include / auto-exclude read-time transform.

    Pure transformation; the service applies it at retrieval (see
    ``TestPolicyRoundTrip`` in ``test_service.py`` for the full write →
    cache-read round-trip).
    """

    def _result(self) -> TaggerResult:
        return TaggerResult(
            tags={
                "general": 0.99,
                "1girl": 0.95,
                "solo": 0.8,
                "long_hair": 0.7,
                "nsfw": 0.6,
            },
            categories={
                "rating": ["general"],
                "general": ["1girl", "solo", "long_hair", "nsfw"],
            },
        )

    def test_empty_policy_returns_input_unchanged(self) -> None:
        """No always_add / banned → identity transform (same object)."""
        result = self._result()
        out = apply_policy(result, TagPolicy())
        assert out is result

    def test_empty_explicit_lists_returns_input_unchanged(self) -> None:
        """Explicit empty lists are equivalent to no policy."""
        result = self._result()
        out = apply_policy(result, TagPolicy(always_add=[], banned=[]))
        assert out is result

    def test_banned_tags_removed_from_tags_and_categories(self) -> None:
        """Banned tag disappears from both ``tags`` and every category."""
        out = apply_policy(self._result(), TagPolicy(banned=["nsfw", "long_hair"]))
        assert "nsfw" not in out.tags
        assert "long_hair" not in out.tags
        assert "nsfw" not in out.categories["general"]
        assert "long_hair" not in out.categories["general"]

    def test_always_add_missing_tag_injected(self) -> None:
        """always_add tag the model missed appears with score 1.0 under general."""
        out = apply_policy(self._result(), TagPolicy(always_add=["masterpiece"]))
        assert out.tags["masterpiece"] == 1.0
        assert "masterpiece" in out.categories["general"]

    def test_always_add_present_tag_keeps_original_score(self) -> None:
        """always_add tag the model produced keeps its real score, no duplicate."""
        out = apply_policy(self._result(), TagPolicy(always_add=["1girl"]))
        assert out.tags["1girl"] == 0.95  # original score, NOT overridden
        assert "1girl" in out.categories["general"]

    def test_always_add_wins_over_banned(self) -> None:
        """Tag in both lists → kept (always_add semantics win)."""
        out = apply_policy(self._result(), TagPolicy(always_add=["nsfw"], banned=["nsfw"]))
        assert out.tags["nsfw"] == 0.6  # present, original score
        assert "nsfw" in out.categories["general"]

    def test_does_not_mutate_input(self) -> None:
        """The input result must be untouched (cache invariants)."""
        result = self._result()
        original_tags = dict(result.tags)
        original_cats = {k: list(v) for k, v in result.categories.items()}
        apply_policy(result, TagPolicy(always_add=["masterpiece"], banned=["nsfw"]))
        assert result.tags == original_tags
        assert result.categories == original_cats

    def test_customizations_passthrough(self) -> None:
        """User-edited customizations ride through unchanged."""
        from yadc.taggers.base import TagCustomizations

        result = self._result()
        result.customizations = TagCustomizations(disabled=["solo"], custom_tags={"general": ["masterpiece"]})
        out = apply_policy(result, TagPolicy(always_add=["masterpiece"], banned=["nsfw"]))
        assert out.customizations is result.customizations
        assert out.customizations.disabled == ["solo"]
        assert out.customizations.custom_tags == {"general": ["masterpiece"]}

    def test_empty_category_after_filtering_dropped(self) -> None:
        """A category that ends up empty after filtering is omitted from output."""
        out = apply_policy(self._result(), TagPolicy(banned=["1girl", "solo", "long_hair", "nsfw"]))
        # 'rating' still has 'general', 'general' is now empty.
        assert "general" not in out.categories
        assert "rating" in out.categories

    def test_multiple_always_adds_with_no_general_category_created(self) -> None:
        """A result with no 'general' category still gets the always_add bucket created."""
        result = TaggerResult(
            tags={"foo": 0.5},
            categories={"character": ["foo"]},
        )
        out = apply_policy(result, TagPolicy(always_add=["masterpiece", "best_quality"]))
        assert out.categories["general"] == ["masterpiece", "best_quality"] or set(out.categories["general"]) == {"masterpiece", "best_quality"}
        assert out.tags["masterpiece"] == 1.0
        assert out.tags["best_quality"] == 1.0


class TestApplyPolicyNormalization:
    """``apply_policy`` membership checks normalise both sides (case
    insensitive, internal whitespace collapsed to underscore) so a
    user-typed ``Speech Bubble`` or ``speech bubble`` resolves to the
    same identity as the canonical model key ``speech_bubble``.
    Storage stays verbatim; only the comparison collapses case and
    whitespace.

    These tests are written so a refactor that drops the normalisation
    (going back to exact-string matching) regresses them clearly.
    """

    def _result(self) -> TaggerResult:
        return TaggerResult(
            tags={
                "1girl": 0.95,
                "speech_bubble": 0.7,
                "long_hair": 0.6,
            },
            categories={
                "general": ["1girl", "speech_bubble", "long_hair"],
            },
        )

    def test_always_add_with_capitals_preserves_model_score(self) -> None:
        """User-typed ``Speech Bubble`` boosts the model's canonical
        ``speech_bubble`` at its original score (no synthetic chip on top)."""
        out = apply_policy(self._result(), TagPolicy(always_add=["Speech Bubble"]))
        assert out.tags["speech_bubble"] == 0.7
        assert "Speech Bubble" not in out.tags
        assert "Speech Bubble" not in out.categories.get("general", [])

    def test_always_add_with_spaces_preserves_model_score(self) -> None:
        """Same as the capitals variant, for the spaces case."""
        out = apply_policy(self._result(), TagPolicy(always_add=["speech bubble"]))
        assert out.tags["speech_bubble"] == 0.7
        assert "speech bubble" not in out.tags
        assert "speech bubble" not in out.categories.get("general", [])

    def test_banned_with_capitals_removes_canonical_tag(self) -> None:
        out = apply_policy(self._result(), TagPolicy(banned=["Speech Bubble"]))
        assert "speech_bubble" not in out.tags
        assert "speech_bubble" not in out.categories.get("general", [])

    def test_banned_with_spaces_removes_canonical_tag(self) -> None:
        out = apply_policy(self._result(), TagPolicy(banned=["speech bubble"]))
        assert "speech_bubble" not in out.tags
        assert "speech_bubble" not in out.categories.get("general", [])

    def test_always_add_wins_over_banned_via_normalisation(self) -> None:
        """Same canonical identity in both lists under different surface
        forms → ``always_add`` wins, tag kept at original score."""
        out = apply_policy(
            self._result(),
            TagPolicy(always_add=["speech bubble"], banned=["Speech Bubble"]),
        )
        assert out.tags["speech_bubble"] == 0.7

    def test_synthetic_injection_skipped_when_norm_matches_model(self) -> None:
        """A user-typed ``Speech Bubble`` doesn't produce a synthetic chip
        on top of an existing canonical ``speech_bubble`` from the model —
        the canonical entry covers it (no duplicate identity)."""
        out = apply_policy(self._result(), TagPolicy(always_add=["Speech Bubble"]))
        # Only one entry for this identity — the model's canonical. The
        # user's verbatim "Speech Bubble" was absorbed into the matching
        # canonical key, not duplicated as a synthetic chip.
        assert "speech_bubble" in out.tags
        assert "Speech Bubble" not in out.tags
        assert "speech bubble" not in out.tags

    def test_synthetic_injection_uses_user_verbatim_for_custom_tag(self) -> None:
        """A genuinely custom tag the model never produced is injected at
        score 1.0 with the user's verbatim identity preserved (storage
        doesn't canonicalise)."""
        out = apply_policy(
            self._result(),
            TagPolicy(always_add=["My Custom Invention"]),
        )
        assert "My Custom Invention" in out.tags
        assert out.tags["My Custom Invention"] == 1.0
        # Casing preserved — the lower-cased canonical form is NOT what
        # ends up in the result. Storage-style verbatim, not normalised.
        assert "my_custom_invention" not in out.tags

    def test_two_user_forms_same_norm_yield_one_synthetic(self) -> None:
        """User types ``my tag`` and ``My Tag`` — same canonical identity.
        Only one synthetic chip entry; the second form collapses onto the
        first (dict-key uniqueness)."""
        result = TaggerResult(tags={"1girl": 0.9}, categories={"general": ["1girl"]})
        out = apply_policy(result, TagPolicy(always_add=["my tag", "My Tag"]))
        # At most one entry exists for this identity — either of the two
        # user-typed forms survives, but never both.
        identity_keys = [t for t in out.tags if t in {"my tag", "My Tag"}]
        assert len(identity_keys) <= 1

    def test_kaomoji_match_across_canonical_forms(self) -> None:
        """A user-curated kaomoji matches the model's identical canonical
        key without writing a synthetic — kaomojis have no whitespace
        and are already lowercase, so the normalisation passes them
        through unchanged."""
        result = TaggerResult(tags={"^_^": 0.7}, categories={"general": ["^_^"]})
        out = apply_policy(result, TagPolicy(always_add=["^_^"]))
        # The model's '^_^' satisfies the always_add entry — preserved
        # at its original score, no synthetic duplicate.
        assert out.tags == {"^_^": 0.7}
