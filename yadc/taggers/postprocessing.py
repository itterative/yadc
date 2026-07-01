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
