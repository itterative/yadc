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
    after thresholding so only surviving tags are touched.
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
    silently dropped). Policy is only applied to **model output** —
    not to a user's interactive prune — so an explicit edit isn't
    second-guessed by a stale policy entry.

    A stdlib dataclass (not a Pydantic model) so the taggers package
    stays free of API-layer concerns; Pydantic still validates it when
    nested as a request-body field (see ``TagJobOptions`` / the
    single-image tag body in the API service).
    """

    always_add: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)


def apply_policy(result: TaggerResult, policy: TagPolicy) -> TaggerResult:
    """Apply the always-add / banned lists to a post-threshold tag result.

    Intended as a read-time transform applied alongside thresholding
    and :func:`replace_underscores`; toggling a tag in either list is
    then reflected on the next read of a cached result without
    re-running the model.

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
    banned = {t for t in policy.banned}
    always = list(policy.always_add)
    if not always and not banned:
        return result

    new_tags: dict[str, float] = {}
    for tag, score in result.tags.items():
        # always_add wins over banned — a tag in both lists is kept
        # at its original score rather than silently removed.
        if tag in banned and tag not in always:
            continue
        new_tags[tag] = score
    new_categories: dict[str, list[str]] = {}
    for cat, tags in result.categories.items():
        kept = [t for t in tags if t not in banned or t in always]
        if kept:
            new_categories[cat] = kept

    # Inject missing always-add tags at score 1.0, filed under
    # ``general``. The synthetic score mirrors how custom user tags
    # score downstream, so formatters / consumers don't need a
    # separate "injected" path.
    general = new_categories.get("general")
    for tag in always:
        if tag in new_tags:
            # Already in the model output — score preserved above.
            continue
        new_tags[tag] = 1.0
        if general is None:
            general = []
            new_categories["general"] = general
        if tag not in general:
            general.append(tag)

    return TaggerResult(tags=new_tags, categories=new_categories, customizations=result.customizations)
