---
date: 2026-06-29
---
# Formatters: tags sorted alphabetically within each section

**Context:** The draft / extras formatters (and the comma formatter feeding
sd-scripts training captions) used to emit tags in the order they appeared
in the model's label file (`selected_tags.csv` row order, which corresponds
to `argsort(scores)` only by coincidence). Re-runs with different thresholds
or different label files produced visibly different orderings, and the
comma-separated training caption was a moving target for diffs.

**Decision:** Sort tags alphabetically within each category, across all three
formatters (`comma`, `structured`, `scored`) and the `extras_tags` helper
that builds the `[tags]` extras sub-table. Section order is unchanged
(rating → character → general).

**Rationale:**

- The user picked alphabetical over probability-descending so re-runs
  produce stable, diff-friendly output (sd-scripts training captions and
  extras TOMLs don't churn between re-tags with the same labels).
- Probability-descending would change outputs across different threshold
  settings even when the underlying tag set is the same, which the user
  specifically didn't want.

**What it changed:**

- `yadc/taggers/formatters.py`:
  - `comma_formatter`, `structured_formatter`, `scored_formatter` all
    sort each section's tag list alphabetically before joining.
  - `extras_tags` sorts `general` / `character` lists alphabetically;
    `rating` stays as a single string (still `top_rating(result)`).
  - `_ordered_tags` (the shared helper) sorts each category list before
    extending; the uncategorized-result branch now sorts
    `result.tags.keys()` rather than relying on dict insertion order.

- `tests/taggers/test_formatters.py`:
  - Existing fixtures happened to be in alphabetical order, so prior
    tests passed unchanged.
  - Added `test_sorts_alphabetically_within_each_section` to all three
    formatter test classes (and `test_uncategorized_result_is_sorted_alphabetically`
    on the comma class) so the ordering guarantee is locked in even if a
    future refactor drops the `sorted(...)` call.
  - Added `TestExtrasTags::test_categories_sorted_alphabetically` covering
    the `general` / `character` sort in the extras sub-table.

- `.pi/agent/memory/docs/tagger-architecture.md`:
  - "formatters" paragraph now mentions alphabetical ordering for
    diff-stable output across re-runs.
  - `extras_tags` note clarifies the alphabetical sort.

**Files touched:** `yadc/taggers/formatters.py`,
`tests/taggers/test_formatters.py`, this entry.
