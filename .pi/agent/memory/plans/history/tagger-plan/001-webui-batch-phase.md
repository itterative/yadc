---
date: 2026-06-28
---
# WebUI surface + batch tagging job

**Context:** The tagger backend (subprocess, lifecycle, single-image endpoint, SSE
events) landed and was verified end-to-end against the `cudlil` dataset. The next
phase is making tagging usable from the WebUI and supporting whole-dataset runs.
The user wants two output destinations — (1) save tags as a **draft** (default
name `tags`), with selectable text formats; (2) save as **extras** under a
`[tags]` sub-table (`tags.general`, `tags.character`, `tags.rating`). The user
wants the **backend** to perform all draft/extras writes (not the frontend), and
wants **one endpoint** that can run the tagger on a single image or the whole
dataset.

**Decision:**

- **Run endpoint (unified, job-based, mirrors captioning):**
  `POST /datasets/<name>/tag` starts a background tagging job.
  `TagJobOptions` carries `image_ids` (optional → all images in the dataset),
  per-category threshold overrides, and optional `TagSaveOptions`. A single
  image is `image_ids: [id]`. Companion routes: `DELETE` (stop), `GET` (status),
  `GET /jobs` (job log). Per-image results flow to clients via the existing
  `ImageTaggedEvent` / `ImageTagErrorEvent` SSE.

- **Save is backend-side and shared.** `TagSaveOptions` on the job controls
  auto-save per image (`mode: none | draft | extras`). The same formatting +
  write helpers serve an explicit interactive save endpoint
  (`POST /datasets/<name>/images/<id>/tags`) that writes user-pruned tags
  without re-running the model.

- **Draft formats are pluggable** (`TagDraftFormatter` keyed by name). Ship two:
  `comma` (default — comma-separated tag list, the sd-scripts training caption)
  and `structured` (semi-structured by category, for feeding Refine). Design
  leaves room for later formats (e.g. weighted-confidence, JSON) without API
  changes — just a new formatter registration.

- **Extras rating is a single top-rating string**, not a list. Rating is
  categorical; storing the highest-rated category as `tags.rating = "general"`
  avoids the "all four ratings pass the 0.0 threshold" problem. `tags.general`
  and `tags.character` stay lists.

- **Extras write is a merge**, not a full replace: a new
  `DatasetService` helper parses the existing per-image TOML, sets/updates the
  `[tags]` sub-table, and re-serializes (preserving other keys). It reuses the
  existing `expect_file_change` watcher-suppression + history-save pattern from
  `update_extras`.

- **Frontend:** new `Tags` tab inside `ImageDetail` (interactive single image:
  Tag → result grid with per-tag confidence + prune → save bar), a new
  `lib/stores/tagging/` domain (mirroring `caption/`), and a batch
  `TagSettingsPanel` mirroring `CaptionSettingsPanel`. SSE zod schemas +
  listener for the three tagger events (currently backend-only).

**Rationale:**

- Job-based run mirrors the proven `CaptioningService` shape (`image_ids`
  parameterization, 202 + SSE), so batch and single image share one code path
  and the frontend reuses the same job-tracking patterns.
- Backend-side writes keep the frontend free of TOML-merge logic and let the
  file watcher's expected-change suppression work uniformly (the backend knows
  the real paths and `source` labels).
- Pluggable draft formatters front-load the extension point the user asked for
  (drop [c] weighted for now, but add later without touching the wire format).
- Top-rating-as-string sidesteps the categorical-vs-list mismatch and matches
  how a refine/template flow would consume a rating.

**Open (need user confirmation before implementation):**

1. Single-image interactive run: keep the current **synchronous** per-image
   endpoint for instant prune-UI population, or make it a **job** (uniform, but
   result arrives via SSE)? Recommendation: synchronous preview keeps the
   interactive UX snappy; batch is the job.
2. Batch panel placement: new top-level side-panel tab ("Tags") vs. folded into
   the existing Caption tab vs. its own dialog.
3. Mobile density: a 4th `ImageDetail` tab (Caption / Preview / Extras / Tags)
   may crowd narrow screens — accepted for now, revisit after build.
