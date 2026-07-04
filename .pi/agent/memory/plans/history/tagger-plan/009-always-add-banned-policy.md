---
date: 2026-07-03
---
# Always-add / banned tag policy

New `TagPolicy` dataclass on `TagJobOptions` + the single-image
body, plus a frontend `tagPolicy` storable + a new "Tags" section in
the TagSettings tab. Lets the user auto-include certain tags
(`always_add`, injected at score 1.0 under `general`) or auto-remove
certain tags (`banned`, removed from both `tags` and every
`categories[cat]`) from every tagged image's result.

Composes cleanly with the existing tier system: the Customize tab's
`starred` / `undesired` members render as quick-add chips in the
Settings tab's two PolicyList instances. The lists themselves are
**separate** storables — a starred tag doesn't have to be in
always_add, and a banned tag doesn't have to be in the undesired
tier. The Customize tab's tier store is unchanged.

Policy is applied as a **read-time transform** via `apply_policy`
inside `_refilter` (thresholds → policy → `replace_underscores`), so
the cached value stays the bare post-threshold model output. Policy
is **not** part of `TaggerResultKey` and **not** baked into the
cached value: a single cache slot serves every policy variant, so
toggling a policy tag shows up on the next read of an existing slot
without re-tagging, and disabling it reverts to the raw model output.
This is deliberate even though the policy is transformative
(`always_add` adds tags; `banned` removes them) — the read-time
application keeps the cache maximally reusable.

`always_add` wins over `banned` for the same name so a stale
accidental collision doesn't silently cancel an always-add entry.

Policy applies **only** to model output — the interactive save
endpoint (`POST .../images/<id>/tags`) writes the user-pruned result
as-is, since the user's explicit prune should win over a stale
policy entry.

The GET cache-read endpoint (`.../images/<id>/tag`) and the preview
endpoint (`.../images/<id>/tags/preview`, for the customization
persistence path) accept the policy as multi-valued query params
(`always_add=tag1&always_add=tag2`); the GET applies it as a
read-time transform over the cached value.

**Files (backend):** `yadc/taggers/postprocessing.py` (`TagPolicy`
dataclass + `apply_policy` pure transform, re-exported from
`yadc/taggers`), `yadc/api/services/tagging.py` (`_refilter` policy
param + service thread-through for `tag_image` / batch loop /
`get_tag_result`), `yadc/api/controllers/api_tagging.py`
(`TagImageBody.policy`, GET query-param handling),
`tests/taggers/test_postprocessing.py` (`TestApplyPolicy` 10 cases),
`tests/taggers/test_service.py` (`TestPolicyRoundTrip` 7 service-level
write→read cases).

**Files (frontend):** `yadc/webui/src/lib/stores/tagging/policy.ts`
new storable + accessors + `snapshotTagPolicy` wire subset,
`yadc/webui/src/lib/stores/tagging/{index,api,actions}.ts` re-export
+ forward the policy on the three request call sites,
`yadc/webui/src/lib/components/tagging/PolicyList.svelte` new
tier-aware chip-row component, `yadc/webui/src/lib/components/tagging/TagSettings.svelte`
new "Tags" section between Thresholds and Save hosting two
`PolicyList` instances.

**Docs:** `tagger-architecture.md` "Tag policy" section + the
`TagPolicy` read-time-transform section;
`frontend/components-domain.md` adds `PolicyList.svelte` and the
new Tags section in TagSettings; `frontend/stores.md` adds
`policy.ts`.
