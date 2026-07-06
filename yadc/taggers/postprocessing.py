from __future__ import annotations

from dataclasses import dataclass, field

from yadc.taggers.base import TaggerResult

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}


def replace_underscore_for_tag(tag: str) -> str:
    """Turn ``_`` into spaces in a single tag, preserving kaomojis.

    Reused by the suggestion endpoint so the autocomplete dropdown can
    mirror the tagger's ``replace_underscores`` setting without duplicating
    the kaomoji allowlist."""
    return tag.replace("_", " ") if tag not in kaomojis else tag


def replace_underscores(result: TaggerResult) -> TaggerResult:
    """Return a new :class:`TaggerResult` with underscores turned into spaces.

    WD-tagger models ship underscored tag names (``1girl``, ``long_hair``);
    training captions read better with spaces. Kaomojis (``^_^``, ``>_<``)
    are preserved — replacing their underscores would corrupt them. Applied
    after thresholding so only surviving tags are touched. This is the
    bare model-output path; curated-list display is a frontend render
    concern and never touches the model result.
    """
    remap = {tag: replace_underscore_for_tag(tag) for tag in result.tags}
    # Only rebuild when something actually changed (the common no-op case
    # returns the original object, so callers can cheaply check identity).
    if not any(new != old for old, new in remap.items()):
        return result
    new_tags = {remap[tag]: score for tag, score in result.tags.items()}
    new_categories = {cat: [replace_underscore_for_tag(t) for t in tags] for cat, tags in result.categories.items()}
    return TaggerResult(tags=new_tags, categories=new_categories)


@dataclass
class TagPolicy:
    """Auto-include / auto-exclude list applied to model-output tag results.

    - ``always_add`` — every entry is forced into the result (score 1.0,
      ``general`` category) regardless of what the model produced. Use
      for tags you want on every tagged image in the dataset.
    - ``banned`` — every entry is removed from the result (every
      ``categories[cat]`` list and the ``tags`` map). Use for tags that
      should never appear regardless of confidence.

    Order is irrelevant in both lists (membership is a set, not a
    sequence). ``always_add`` wins over ``banned`` when the same name
    appears in both (otherwise the always-add semantics would be
    silently dropped).
    """

    always_add: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)


def _norm_key(tag: str) -> str:
    """Normalize a tag string for *membership* checks in :func:`apply_policy`.

    Storage (the backend ``always_add`` / ``banned`` lists and the model's
    output keys) stays verbatim — this helper only collapses the
    comparison key so a user-typed ``Speech Bubble`` resolves to the
    same identity as the model's canonical ``speech_bubble``.

    Behaviour:

    - ``strip`` trims surrounding whitespace before any other transform.
    - ``lower`` gives the case-insensitive comparison the danbooru-
      trained tagger models already produce (their keys are lowercase).
    - ``split()`` + ``'_'.join(...)`` collapses any whitespace run
      between words to a single ``_`` (``'foo bar'`` → ``'foo_bar'``,
      ``'a   b\\tc'`` → ``'a_b_c'``).
    - **Idempotent and lossy**: running it twice is the same as once;
      case is dropped.
    - **Kaomoji-safe**: ``^_^``, ``(o)_(o)``, ``>_<`` etc. contain no
      whitespace and are already lowercase, so they're a no-op pass
      through.

    Distinct from :func:`replace_underscores` — that's a display-time
    transform on the model output (case preserved); this is a lossy
    comparison key for curated-list membership.
    """
    return "_".join(tag.strip().lower().split())


def apply_policy(result: TaggerResult, policy: TagPolicy) -> TaggerResult:
    """Apply the always-add / banned lists to a post-threshold tag result.

    Intended as a read-time transform applied alongside thresholding
    and :func:`replace_underscores`; toggling a tag in either list is
    then reflected on the next read of a cached result without
    re-running the model.

    Membership checks compare against a per-tag *normalized key*
    (see :func:`_norm_key`) rather than the raw string. The model
    output's tag keys are canonical-lowercase-underscored
    (``speech_bubble``); the user-curated lists may contain anything
    the user typed (``Speech Bubble``, ``speech bubble``, etc.).
    Normalizing both sides on read means a user-typed ``Speech Bubble``
    in ``always_add`` does what the user intended (boosts the model's
    canonical ``speech_bubble``) without forcing the stored value to
    be canonicalized at insert time. Storage stays verbatim; only the
    comparison collapses case and whitespace differences.

    Always-add tags the model already produced are kept at their
    original score (no duplicate ``1.0`` override); missing ones are
    injected with score ``1.0`` into the ``general`` category (there's
    no other natural home for a forced-include that the model never
    saw). ``always_add`` wins over ``banned`` for the same name, so a
    user-facing list that ends up in both fields by accident still
    gets the intent of "always have it". ``rating`` is categorical
    metadata; an ``always_add`` tag would slot into ``general``
    regardless, since there's no scenario where the model and user
    tagger differ on rating and the policy should override one to
    match the other.

    Returns a shallow copy so the caller's value stays untouched.
    User-edited customizations ride through unchanged; the policy is
    about what the model output, not what the user pruned.
    """
    if not policy.always_add and not policy.banned:
        return result

    banned = {_norm_key(t) for t in policy.banned}
    always_norm = {_norm_key(t) for t in policy.always_add}

    new_tags: dict[str, float] = {}
    for tag, score in result.tags.items():
        # always_add wins over banned — a tag in both lists is kept
        # at its original score rather than silently removed.
        norm = _norm_key(tag)
        if norm in banned and norm not in always_norm:
            continue
        new_tags[tag] = score
    new_categories: dict[str, list[str]] = {}
    for cat, tags in result.categories.items():
        kept = [t for t in tags if (norm := _norm_key(t)) not in banned or norm in always_norm]
        if kept:
            new_categories[cat] = kept

    # Inject missing always-add tags at score 1.0, filed under ``general``.
    # The dedup check uses the same normalized key as above so a user
    # typed ``Speech Bubble`` doesn't produce a synthetic chip on top of
    # an existing canonical ``speech_bubble`` from the model — instead,
    # the canonical entry is kept (with its model score) and the user's
    # intent is satisfied by the model tag itself. Synthetic entries
    # only appear for genuinely custom tags the model never produced;
    # those keep the user's verbatim identity so the chip surfaces
    # what they typed rather than a rewritten form.
    general = new_categories.get("general")
    seen_norms = {_norm_key(t) for t in new_tags}
    for tag in policy.always_add:
        if _norm_key(tag) in seen_norms:
            # Already covered by an existing entry (model or earlier
            # synthetic). Skip the redundant inject; the canonical key
            # keeps the original score intact.
            continue
        new_tags[tag] = 1.0
        seen_norms.add(_norm_key(tag))
        if general is None:
            general = []
            new_categories["general"] = general
        if tag not in general:
            general.append(tag)

    return TaggerResult(tags=new_tags, categories=new_categories, customizations=result.customizations)
