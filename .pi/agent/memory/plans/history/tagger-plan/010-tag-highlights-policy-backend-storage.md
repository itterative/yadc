---
date: 2026-07-05
---
# Backend storage for tag highlights + per-dataset policy

The Phase I tiers (`tagHighlights`) and Phase J policy
(`tagPolicy`) ships as **localStorage-backed frontend stores** —
the same shape as `tagSettings` and `recentTags`. The policy is
already applied server-side via `apply_policy` (Phase J), but the
lists themselves live only on the client and are snapshotted into
every request body. Multi-device / multi-tab scenarios don't
share them, and there's no audit trail of which dataset the user
configured which policy on. Lift both stores to backend storage.

## Decisions

- **Highlights are global.** The three tiers
  (`starred` / `desired` / `undesired`) plus `categoryOverrides`
  form a single user-curated vocabulary reused across every
  dataset — same shape, no per-dataset variant. Persist in the
  existing `settings` KV table under key `tagger.tag_highlights`,
  mirroring `tagger.active_model` and `tagger.suggestion_variant`.

- **Policy is per-dataset.** `always_add` / `banned` are
  dataset-specific curation (e.g. "always tag this style" vs.
  "never tag this content") — global would be wrong. New table
  `dataset_settings` shaped exactly like `settings` but
  scoped by `dataset_id`:

  ```sql
  CREATE TABLE IF NOT EXISTS dataset_settings (
      dataset_id INTEGER NOT NULL,
      key        TEXT    NOT NULL,
      value      TEXT    NOT NULL,
      updated_t  REAL    NOT NULL DEFAULT (unixepoch()),
      PRIMARY KEY (dataset_id, key),
      FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
  );
  CREATE INDEX IF NOT EXISTS idx_dataset_settings_dataset
      ON dataset_settings (dataset_id);
  ```

  Two rows per dataset today (`policy_always_add`,
  `policy_banned`); a missing row decodes as `[]` so a never-touched
  dataset has the same behaviour as an empty policy. CASCADE on
  dataset delete removes both rows automatically.

  The KV shape (rather than a wide row with both lists as columns)
  follows the existing `settings` convention and keeps room for
  future per-dataset settings without another migration.

- **Drop the body fields.** `TagImageBody` and `TagJobOptions`
  no longer carry `always_add` / `banned` — the backend resolves
  the policy from `dataset_settings` for the request's dataset.
  The frontend `tagPolicy` Svelte store stays as a mirror so the
  PolicyList UI is still reactive; mutators update locally + PUT
  to the new endpoint. `apply_policy` / `_refilter` are unchanged
  (policy remains a read-time transform on the cached result).
  When the local policy changes, the active image's cached
  `TaggerResult` is invalidated so the next read re-applies the
  new policy — same hook already needed for the existing
  customization persistence path.

## Scope

- In scope: `tagHighlights` (global) + `tagPolicy` (per-dataset).
- Out of scope: `tagSettings` (thresholds + save mode) and
  `recentTags` (recent custom-tag autocomplete history) stay in
  localStorage.

## Implementation sketch

**Backend**
- `yadc/api/migrations/0009_dataset_settings_{up,down}.sql` —
  new table + index; down drops both. SQL-only (no Python hook).
- `yadc/api/services/dataset_settings_repository.py` —
  `SettingsRepository`-shaped repo accepting `(dataset_id, key)`:
  `get(dataset_id, key) -> str | None`, `upsert(...)`,
  `delete(...)`, `list_for_dataset(...)`. JSON in/out at the
  service boundary.
- `yadc/api/services/tag_policy_service.py` — wraps the repo,
  exposes `get(dataset_name) -> TagPolicy` (empty policy when no
  rows) and `set(dataset_name, policy)`. Resolves `dataset_id`
  via `DatasetRepository.get_dataset_row` so the public API is
  dataset-name-keyed.
- `yadc/api/services/tag_highlights_service.py` — thin wrapper
  over `SettingsService` with key `tagger.tag_highlights`. Reads
  decode back into the existing `TagHighlights` interface shape
  (we'd keep the same keys: `starred` / `desired` / `undesired` /
  `categoryOverrides`).
- `yadc/api/services/tagging.py` — inject the policy service.
  Remove `policy: TagPolicy | None` from `tag_single_image_async`,
  `start_tag_job_async`, `get_tag_result`, and
  `preview_image_tags`. Resolve per call (or once per batch job
  via a request-scoped cache to avoid per-image reads).
- `yadc/api/controllers/api_tagging.py` — drop
  `TagImageBody.policy`; drop `TagJobOptions.policy`; add:
  - `GET  /tagging/highlights`
  - `PUT  /tagging/highlights` (body = full `TagHighlights`,
    validated, persisted whole-blob)
  - `GET  /datasets/<name>/tag/policy` — 404 if dataset
    unregistered.
  - `PUT  /datasets/<name>/tag/policy` — body
    `{always_add, banned}`.

**Frontend**
- `yadc/webui/src/lib/stores/tagging/highlights.ts` — replace
  the `storable()` primitive with a `writable` store seeded from
  `fetchTagHighlights()`. Mutations (`setTagTier`,
  `removeTagTier`, `clearTier`, `setTagCategoryOverride`,
  `removeTagCategoryOverride`) update the local store
  optimistically and PUT the full payload. Module-level single
  fetch promise so multiple subscribers don't re-fetch on boot.
- `yadc/webui/src/lib/stores/tagging/policy.ts` — per-dataset
  mirror store. New `loadPolicy(datasetName)` helper called from
  the dataset page effect. Mutators update locally + PUT.
- `yadc/webui/src/lib/stores/tagging/api.ts` — add
  `fetchTagHighlights`, `putTagHighlights`, `fetchTagPolicy`,
  `putTagPolicy`; drop `always_add`/`banned` from
  `tagImage`/`startTagJob`/`fetchTagResult`/`previewImageTags`
  request bodies.
- `yadc/webui/src/lib/stores/tagging/actions.ts` — drop
  `snapshotTagPolicy()`; `TagOptions` loses `always_add` /
  `banned`.
- `yadc/webui/src/lib/stores/tagging/results.ts` — add a
  subscription from `tagPolicy` so an updated policy invalidates
  the active dataset's cached `TaggerResult`s (they re-fetch on
  the next viewer-driven read and re-apply the new policy).
- App boot / dataset switch: trigger the initial fetches
  (`+layout.svelte` for highlights, `routes/datasets/[name]/+page.svelte`
  for policy).

**Tests**
- `tests/api/services/test_dataset_settings_repository.py`
  — CRUD + JSON round-trip + cascade on dataset delete.
- `tests/api/services/test_tag_policy_service.py` —
  defaults when no row, round-trip, isolation between
  datasets.
- `tests/api/services/test_tag_highlights_service.py` —
  defaults, round-trip.
- `tests/api/controllers/test_tagging_highlights.py` and
  `test_tagging_policy.py` — endpoint shape, 404 for
  un-registered dataset.
- Update any frontend store / action tests for the
  fetch-backed shapes.

## Migration of existing data

Feature is unreleased; drop the localStorage blobs on first
load of the new stores (a no-op for users since there's
nothing to migrate). `$version` field on `tagHighlights` is
no longer relevant at the wire boundary — drop it cleanly.

## Out of scope

- TOML config integration. `dataset_settings` is a separate
  table, not a section of the existing TOML — keeps the
  picture pure: TOML = caption settings, `settings`/`dataset_settings`
  = user / dataset preferences, `dataset_images` / extras =
  per-image state.
- Backend cache keying for policy. The cache key already
  excludes policy (Phase J); backend reads of the per-dataset
  row happen at the API edge, not in the cache layer.
