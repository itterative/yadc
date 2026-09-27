import { API_BASE, apiErrorMessage } from '$lib/api';
import { clientId } from '../events';
import type {
    ActiveTaggerResponse,
    CancelResult,
    SuggestionVariantResponse,
    SwapTaggerBody,
    SwapTaggerResponse,
    TagJobInfo,
    TaggerModelSummary,
    TaggerResult,
    TagCustomizations,
    TagSaveOptions,
    TagSuggestion
} from './types';

// --- Tagging API helpers ---
//
// The tagger endpoints do not require a password by default, so these are
// plain fetches (no ``withPasswordRetry``). Errors surface as toasts at the
// action layer — including a 503 "tagger not configured", whose message is
// readable enough to show directly.

/** Tag a single image synchronously. Returns the thresholded result
 *  immediately (used by the interactive Tag button — no job/SSE round-trip).
 *  503 when the tagger isn't configured. The always-add / banned policy
 *  is resolved server-side per dataset (see :mod:`yadc.api.services.tag_policy_service`)
 *  — the request body no longer carries it. */
export async function tagImage(
    datasetName: string,
    imageId: number,
    options: {
        rating_threshold?: number;
        general_threshold?: number;
        character_threshold?: number;
        replace_underscores?: boolean;
        per_tag_thresholds?: boolean;
        per_tag_column?: string;
        source?: string;
    } = {},
    signal?: AbortSignal
): Promise<TaggerResult> {
    const params = new URLSearchParams();
    if (options.source) {
        params.set('source', options.source);
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tag${qs ? '?' + qs : ''}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rating_threshold: options.rating_threshold,
                general_threshold: options.general_threshold,
                character_threshold: options.character_threshold,
                replace_underscores: options.replace_underscores,
                per_tag_thresholds: options.per_tag_thresholds,
                per_tag_column: options.per_tag_column
            }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Start a batch tagging job. Returns initial job info (``202``).
 *  ``imageIds`` omitted/empty → whole dataset; otherwise just those.
 *  Thresholds fall back to the server config when undefined. The
 *  always-add / banned policy is resolved server-side per dataset —
 *  the request body no longer carries it. */
export async function startTagJob(
    datasetName: string,
    options: {
        image_ids?: number[];
        rating_threshold?: number;
        general_threshold?: number;
        character_threshold?: number;
        replace_underscores?: boolean;
        per_tag_thresholds?: boolean;
        per_tag_column?: string;
        save?: TagSaveOptions;
        source?: string;
    } = {},
    signal?: AbortSignal
): Promise<TagJobInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Stop a running tagging job. Raw API call — for toast-enabled version
 *  use ``taggingActions.stopTagging``. */
export async function stopTagJob(datasetName: string, signal?: AbortSignal): Promise<boolean> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag`, {
        method: 'DELETE',
        signal
    });
    return res.ok;
}

/** Cancel a tagging job and/or interrupt an in-flight tag (graceful-first,
 *  kill-as-fallback). Pass a ``jobId`` to cancel a specific batch job;
 *  omit it for the single-image case (cancel escalates straight to killing
 *  the subprocess if a request is mid-inference). Returns the outcome. */
export async function cancelTagging(jobId?: string, signal?: AbortSignal): Promise<CancelResult> {
    const res = await fetch(`${API_BASE}/api/tagging/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId ?? null }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Fetch the current tagging job status for a dataset. */
export async function fetchTagJobStatus(
    datasetName: string,
    signal?: AbortSignal
): Promise<TagJobInfo> {
    const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag`, {
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Write user-pruned tags for an image to a draft / extras without
 *  re-running the model. Used by the interactive Tags tab's save bar.
 *  The body echoes the ``TaggerResult`` shape so the backend persists
 *  exactly what the user kept after pruning. */
export async function saveImageTags(
    datasetName: string,
    imageId: number,
    result: TaggerResult,
    save: TagSaveOptions,
    signal?: AbortSignal
): Promise<void> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tags?source=${encodeURIComponent(clientId)}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tags: result.tags, categories: result.categories, save }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}

/** Read the cached tag result for an image from the backend LRU.
 *  Returns ``null`` when no entry exists (never tagged, evicted by LRU,
 *  or tagged under settings that no longer match the current config).
 *  Used by the interactive Tags tab on a fresh page load to surface
 *  results produced by an earlier batch job without re-running the
 *  model. The thresholds options are passed through to the backend so
 *  the GET lands in the same cache bucket as the original POST /tag
 *  write. ``replace_underscores`` is a read-time transform (not part
 *  of the cache key) so it applies on top of whatever the original
 *  write cached. The always-add / banned policy is resolved
 *  server-side per dataset (see :mod:`policy.ts`); a separate fetch /
 *  refetch is needed to surface a policy change, since the cache slot
 *  itself is unchanged. */
export async function fetchTagResult(
    datasetName: string,
    imageId: number,
    options: {
        rating_threshold?: number | null;
        general_threshold?: number | null;
        character_threshold?: number | null;
        replace_underscores?: boolean | null;
        per_tag_thresholds?: boolean | null;
        per_tag_column?: string | null;
    } = {},
    signal?: AbortSignal
): Promise<TaggerResult | null> {
    const params = new URLSearchParams();
    if (options.rating_threshold != null) {
        params.set('rating_threshold', String(options.rating_threshold));
    }
    if (options.general_threshold != null) {
        params.set('general_threshold', String(options.general_threshold));
    }
    if (options.character_threshold != null) {
        params.set('character_threshold', String(options.character_threshold));
    }
    if (options.replace_underscores != null) {
        params.set('replace_underscores', options.replace_underscores ? 'true' : 'false');
    }
    if (options.per_tag_thresholds != null) {
        params.set('per_tag_thresholds', options.per_tag_thresholds ? 'true' : 'false');
    }
    if (options.per_tag_column != null) {
        params.set('per_tag_column', options.per_tag_column);
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tag${qs ? '?' + qs : ''}`,
        { signal }
    );
    if (res.status === 404) {
        return null;
    }
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TaggerResult;
}

/** Preview what :func:`saveImageTags` would write, without touching disk.
 *  Returns formatter text (draft) / the ``[tags]`` TOML sub-table (extras) /
 *  empty string (none). When ``customizations`` is supplied it is also
 *  persisted onto the backend's cached result for this image (same
 *  cache bucket as the threshold options) so navigation restores the
 *  selection. The threshold options are forwarded as query params so
 *  the persist lands in the same bucket as the original POST /tag. */
export async function previewImageTags(
    datasetName: string,
    imageId: number,
    result: TaggerResult,
    save: TagSaveOptions,
    options: {
        rating_threshold?: number | null;
        general_threshold?: number | null;
        character_threshold?: number | null;
        per_tag_thresholds?: boolean | null;
        per_tag_column?: string | null;
        customizations?: TagCustomizations | null;
    } = {},
    signal?: AbortSignal
): Promise<string> {
    const params = new URLSearchParams();
    if (options.rating_threshold != null) {
        params.set('rating_threshold', String(options.rating_threshold));
    }
    if (options.general_threshold != null) {
        params.set('general_threshold', String(options.general_threshold));
    }
    if (options.character_threshold != null) {
        params.set('character_threshold', String(options.character_threshold));
    }
    if (options.per_tag_thresholds != null) {
        params.set('per_tag_thresholds', options.per_tag_thresholds ? 'true' : 'false');
    }
    if (options.per_tag_column != null) {
        params.set('per_tag_column', options.per_tag_column);
    }
    const qs = params.toString();
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/tags/preview${qs ? '?' + qs : ''}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tags: result.tags,
                categories: result.categories,
                save,
                customizations: options.customizations
                    ? {
                          disabled: options.customizations.disabled,
                          custom_tags: options.customizations.custom_tags
                      }
                    : null
            }),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = (await res.json()) as { content: string };
    return data.content;
}

/** Context-free format preview — no dataset or image required. Used by
 *  settings panels to show what a save produces from fixed mock data.
 *  Same return shape as :func:`previewImageTags`. */
export async function previewTagFormats(
    result: TaggerResult,
    save: TagSaveOptions,
    signal?: AbortSignal
): Promise<string> {
    const res = await fetch(`${API_BASE}/api/tagging/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags: result.tags, categories: result.categories, save }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = (await res.json()) as { content: string };
    return data.content;
}

// --- Tag highlights (global tier curation) ---
//
// ``/api/tagging/highlights`` is a single-blob key under the
// server-side ``settings`` KV table at ``tagger.tag_highlights``. The
// shape mirrors the persisted :class:`yadc.api.services.tag_highlights_service.TagHighlights`
// so the round-trip is a pure identity. Mirrors how ``tagger.active_model``
// and ``tagger.suggestion_variant`` are exposed.

/** One curated-tier entry on the wire.

 *  ``name`` is the canonical identity (``speech_bubble``) — stored and
 *  returned verbatim; the frontend projects it to the user's preferred
 *  display form at render time. ``canonical_form`` is ``true`` for
 *  catalog / model-output entries (the frontend applies the user's
 *  ``replace_underscores`` preference when rendering); ``false`` for
 *  free-text user input or kaomojis (rendered verbatim). The backend
 *  validator auto-flips the flag to ``false`` for kaomojis, so the
 *  frontend never has to special-case them. */
export interface TaggedEntryPayload {
    name: string;
    canonical_form: boolean;
}

/** Wire shape returned by :func:`fetchTagHighlights` (and accepted by
 *  :func:`putTagHighlights`). Matches the backend ``TagHighlights``
 *  dataclass field-for-field; tier entries carry the display-form
 *  signal via :class:`TaggedEntryPayload`. */
export interface TagHighlightsPayload {
    starred: TaggedEntryPayload[];
    desired: TaggedEntryPayload[];
    undesired: TaggedEntryPayload[];
    category_overrides: Record<string, string>;
}

/** Fetch the persisted global tag highlights. Returns the same shape
 *  regardless of whether the user has stored anything yet (a fresh
 *  install is an all-empty object). Each tier entry is the canonical
 *  ``{name, custom}`` identity — the frontend applies the user's
 *  ``replaceUnderscores`` preference at render time. Throws on a
 *  non-200 — the action layer typically toasts the failure rather than
 *  handling it explicitly. */
export async function fetchTagHighlights(signal?: AbortSignal): Promise<TagHighlightsPayload> {
    const res = await fetch(`${API_BASE}/api/tagging/highlights`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TagHighlightsPayload;
}

/** Replace the persisted tag highlights with ``value`` (whole-blob write).

 *  Returns the persisted canonical shape so the frontend mirror stays in
 *  sync without a local transform. 400 on Pydantic validation failure
 *  (e.g. a wrong shape). The frontend always sends the complete current
 *  state — partial updates aren't worth a merge protocol here. */
export async function putTagHighlights(
    value: TagHighlightsPayload,
    signal?: AbortSignal
): Promise<TagHighlightsPayload> {
    const res = await fetch(`${API_BASE}/api/tagging/highlights`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TagHighlightsPayload;
}

// --- Dataset tag policy (always_add / banned) ---
//
// ``GET/PUT /api/datasets/<name>/tag/policy``. Per-dataset; key
// ``policy_always_add`` / ``policy_banned`` in the new
// ``dataset_settings`` SQL table. The frontend mirror store caches
// the same object so the UI is reactive; mutators update locally +
// PUT. 404 when the dataset isn't registered.

/** Wire shape returned by :func:`fetchTagPolicy`. Matches the backend
 *  ``StoredPolicy`` field names (``always_add`` / ``banned``,
 *  snake_case for consistency with the rest of the wire surface).
 *  Each list is a ``TaggedEntryPayload[]`` so the ``canonical_form``
 *  display-form signal survives the round-trip. */
export interface TagPolicyPayload {
    always_add: TaggedEntryPayload[];
    banned: TaggedEntryPayload[];
}

/** Fetch the persisted always-add / banned policy for ``datasetName``.

 *  Each entry is the canonical ``{name, canonical_form}`` identity
 *  (the frontend projects to display at render time). 404 when the
 *  dataset is not registered with yadc; action layer surfaces the
 *  toast. */
export async function fetchTagPolicy(
    datasetName: string,
    signal?: AbortSignal
): Promise<TagPolicyPayload> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag/policy`,
        { signal }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TagPolicyPayload;
}

/** Persist the always-add / banned policy for ``datasetName`` (whole-blob write).

 *  Returns the persisted canonical shape so the frontend mirror stays in
 *  sync without a local transform. 404 when the dataset is not registered
 *  (the action layer calls :func:`fetchTagPolicy` first to verify
 *  registration); 400 on Pydantic validation failure. Empty lists are
 *  persisted normally — clearing a list is a real edit. */
export async function putTagPolicy(
    datasetName: string,
    value: TagPolicyPayload,
    signal?: AbortSignal
): Promise<TagPolicyPayload> {
    const res = await fetch(
        `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/tag/policy`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(value),
            signal
        }
    );
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TagPolicyPayload;
}

// --- Tagger model swap ---

/** Fetch the persisted active tagger selection + subprocess liveness.
 *  Used by the SettingsDialog picker to show "Currently running: …". */
export async function fetchActiveTagger(signal?: AbortSignal): Promise<ActiveTaggerResponse> {
    const res = await fetch(`${API_BASE}/api/tagger/active`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as ActiveTaggerResponse;
}

export interface TaggerModelsResponse {
    models: TaggerModelSummary[];
    local: TaggerModelSummary;
    /** Names of supported preproc profiles (mirrors
     *  ``yadc.taggers.onnx_preprocess.list_profiles``). Drives the
     *  Profile dropdown in the picker. */
    profiles: string[];
}

/** Fetch the curated model catalog (SmilingWolf HF repos + the local-file sentinel).
 *  The picker dropdown renders from this; the user picks one and we POST to
 *  ``swapTaggerModel`` with a fully-formed selection. */
export async function listTaggerModels(signal?: AbortSignal): Promise<TaggerModelsResponse> {
    const res = await fetch(`${API_BASE}/api/tagger/models`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as TaggerModelsResponse;
}

// --- Tag suggestion autocomplete (Tags tab custom-tag input) ---
//
// Session-scoped LRU keyed by normalized query string: backspace-and-retype
// within the same keystroke sequence re-serves cached results instead of
// re-fetching. 64-entry insertion-order Map; oldest key evicted on overflow.

const SUGGESTION_CACHE_LIMIT = 64;
const suggestionCache = new Map<string, TagSuggestion[]>();

/** Move the key to the most-recently-used position and return its value. */
function lruGet(key: string): TagSuggestion[] | undefined {
    const value = suggestionCache.get(key);
    if (value === undefined) {
        return undefined;
    }
    suggestionCache.delete(key);
    suggestionCache.set(key, value);
    return value;
}

/** Insert (or refresh) a key, evicting the oldest if over capacity. */
function lruSet(key: string, value: TagSuggestion[]) {
    if (!suggestionCache.has(key) && suggestionCache.size >= SUGGESTION_CACHE_LIMIT) {
        const oldest = suggestionCache.keys().next().value;
        if (oldest !== undefined) {
            suggestionCache.delete(oldest);
        }
    }
    suggestionCache.set(key, value);
}

/** Fetch autocomplete suggestions for *query*. Each suggestion carries its
 *  catalog category so the dropdown can show a category badge. Names come
 *  back in canonical form (``speech_bubble``); the dropdown projects to the
 *  user's display preference at render time. Cached by normalized query
 *  (+ limit) so backspace-and-retype within the session is instant; empty
 *  results aren't cached (they say nothing about a longer query that
 *  extends them). */
export async function fetchTagSuggestions(
    query: string,
    signal?: AbortSignal,
    limit = 20
): Promise<TagSuggestion[]> {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
        return [];
    }

    const cacheKey = `${limit}:${normalized}`;
    const cached = lruGet(cacheKey);
    if (cached) {
        return cached;
    }

    const params = new URLSearchParams({
        q: query.trim(),
        limit: String(limit)
    });
    const res = await fetch(`${API_BASE}/api/tagging/suggest?${params}`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    const data = (await res.json()) as { query: string; suggestions: TagSuggestion[] };
    if (data.suggestions.length > 0) {
        lruSet(cacheKey, data.suggestions);
    }
    return data.suggestions;
}

// --- Tag suggestion variant (catalog selection) ---

/** Fetch the active suggestion variant, the config default, and the full
 *  variant list in one payload. Used by the Settings picker to populate
 *  the dropdown and mark the current selection. Unlike the tagger swap,
 *  this has no busy/in-progress states — the backend persists and kicks a
 *  background reload, returning immediately. */
export async function fetchSuggestionVariant(
    signal?: AbortSignal
): Promise<SuggestionVariantResponse> {
    const res = await fetch(`${API_BASE}/api/tagging/suggest/variant`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as SuggestionVariantResponse;
}

/** Persist a new active suggestion variant. The backend coerces the value
 *  to the enum (400 on an unknown value) and drops its cached catalog; the
 *  new variant downloads on the next autocomplete request. Throws on a
 *  non-200 so the action layer can toast. */
export async function setSuggestionVariant(
    variant: string,
    signal?: AbortSignal
): Promise<SuggestionVariantResponse> {
    const res = await fetch(`${API_BASE}/api/tagging/suggest/variant`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variant }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return (await res.json()) as SuggestionVariantResponse;
}

/** Discriminated result from :func:`swapTaggerModel`. The action layer
 *  matches on ``status`` to pick the right toast (and never has to
 *  re-parse the response body or distinguish 409 from 429). */
export type SwapTaggerResult =
    | { status: 'ok'; response: SwapTaggerResponse }
    | { status: 'busy'; message: string }
    | { status: 'in_progress'; message: string; retryAfterS: number }
    | { status: 'error'; message: string };

/** Swap the active tagger selection. Maps the backend's status codes into a
 *  small set of variants the UI can switch on without re-parsing the body:
 *  200 → ``ok``; 409 → ``busy`` (batch running); 429 → ``in_progress``
 *  (concurrent swap, with retry hint); anything else → ``error``. */
export async function swapTaggerModel(
    body: SwapTaggerBody,
    signal?: AbortSignal
): Promise<SwapTaggerResult> {
    const res = await fetch(`${API_BASE}/api/tagger/swap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal
    });
    if (res.ok) {
        const response = (await res.json()) as SwapTaggerResponse;
        return { status: 'ok', response };
    }
    let parsed: { error?: string; retry_after_s?: number } | null = null;
    try {
        parsed = (await res.json()) as { error?: string; retry_after_s?: number };
    } catch {
        /* non-JSON body */
    }
    const message = parsed?.error || `Swap failed (HTTP ${res.status})`;
    if (res.status === 409) {
        return { status: 'busy', message };
    }
    if (res.status === 429) {
        return { status: 'in_progress', message, retryAfterS: parsed?.retry_after_s ?? 2 };
    }
    return { status: 'error', message };
}
