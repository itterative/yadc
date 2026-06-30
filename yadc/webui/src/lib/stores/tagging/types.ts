// Types matching the backend tagger API dataclasses / SSE events.

/** Per-image tag result — mirrors ``TaggerResult`` (``yadc/taggers/base.py``)
 *  and the JSON the synchronous ``tag`` endpoint returns. */
export interface TaggerResult {
    /** All detected tags with their scores (0–1). Every tag listed in
     *  ``categories`` also appears here. */
    tags: Record<string, number>;
    /** Tags grouped by category (e.g. ``{rating, general, character}``). */
    categories: Record<string, string[]>;
}

/** Where / how a tag result is persisted for an image. Mirrors
 *  ``TagSaveOptions`` (``yadc/api/services/tagging.py``). */
export interface TagSaveOptions {
    /** ``none`` = tag only; ``draft`` = formatted text to a draft sidecar;
     *  ``extras`` = merge a ``[tags]`` sub-table into the image TOML. */
    mode: 'none' | 'draft' | 'extras';
    /** Draft name when ``mode === 'draft'``. */
    draft_name: string;
    /** Registered ``TagDraftFormatter`` name when ``mode === 'draft'``. */
    draft_format: string;
    /** Batch-only: when ``false`` (default), images that already carry the
     *  target artifact (a draft of the same name, or a ``[tags]`` sub-table)
     *  are skipped entirely. The interactive save endpoint always writes. */
    overwrite: boolean;
}

/** Snapshot of a running (or recently finished) batch tagging job.
 *  Mirrors ``TagJobInfo``. */
export interface TagJobInfo {
    status: 'idle' | 'running' | 'stopping' | 'error' | 'done' | 'cancelled';
    dataset_name: string;
    job_id: string;
    processed: number;
    total: number;
    errors: number;
    error: string | null;
    error_messages: string[];
    /** Frontend-supplied / fallback model label. */
    source: string;
    /** Seconds since the job started. */
    elapsed: number;
}

/** Lifecycle of the tagger subprocess (global — one subprocess).
 *  Mirrors ``TaggerStatusEvent``. */
export interface TaggerStatus {
    state: 'starting' | 'ready' | 'stopping' | 'stopped' | 'failed';
    source: string;
    error: string | null;
}

/** Outcome of a cancel request. Mirrors ``CancelResult`` (backend).
 *  ``killed`` means the subprocess was force-terminated (an inference
 *  was in flight past the grace window); ``stopped`` means a job's
 *  stop_event was set / work finished without a kill; ``stale_job`` /
 *  ``nothing_running`` mean there was nothing to cancel. */
export interface CancelResult {
    outcome: 'stopped' | 'killed' | 'stale_job' | 'nothing_running';
    killed: boolean;
    job_id: string;
}

/** Default ``TagSaveOptions``. */
export const DEFAULT_SAVE: TagSaveOptions = {
    mode: 'draft',
    draft_name: 'tags',
    draft_format: 'comma',
    overwrite: false
};

// --- SSE event types (field shapes, validated by zod in events.ts) ---

export interface TaggerStatusEvent {
    state: TaggerStatus['state'];
    source: string;
    error: string | null;
}

export interface ImageTaggedEvent {
    dataset_name: string;
    image_id: number;
    file_name: string;
    path: string;
    tags: Record<string, number>;
    categories: Record<string, string[]>;
    source: string;
    duration_ms: number;
}

export interface ImageTagErrorEvent {
    dataset_name: string;
    image_id: number;
    error: string;
    source: string;
    duration_ms: number;
}

export interface TagJobStatusEvent {
    status: TagJobInfo['status'];
    dataset_name: string;
    processed: number;
    total: number;
    errors: number;
    job_id: string;
    error: string | null;
    source: string;
    elapsed: number;
}

// --- Tagger model swap (mirrors ActiveTagger / TaggerModelSummary on the backend) ---

/** One entry in the curated model catalog surfaced by ``GET /api/tagger/models``. */
export interface TaggerModelSummary {
    id: string;
    display: string;
    params: string;
    /** Profile applied when this row is swapped. For curated HF models this
     *  is baked in (``"wd-tagger"`` for SmilingWolf) and the picker hides the
     *  Profile control. For the Local sentinel it seeds the picker's Profile
     *  dropdown when no active selection exists to copy from. */
    default_preproc_profile: string;
    /** Override input size for symbolic-dim models. ``0`` = use the model's
     *  default. Only meaningful for Local selections — curated HF rows use
     *  ``0`` (their native size). */
    default_size: number;
}

/** The persisted active-tagger selection (round-trips through ``POST /api/tagger/swap``). */
export interface ActiveTaggerSelection {
    kind: 'hf' | 'local';
    repo_id: string;
    repo_model_filename: string;
    repo_label_filename: string;
    model_path: string;
    label_path: string;
    preproc_profile: string;
    default_size: number;
    /** Server-derived source label (``hf:<repo_id>`` / ``local:<path>``). */
    source: string;
}

/** ``GET /api/tagger/active`` response shape. ``active === null`` when nothing is configured. */
export interface ActiveTaggerResponse {
    active: ActiveTaggerSelection | null;
    is_available: boolean;
}

/** Body for the swap request — ``ActiveTaggerSelection`` minus the server-derived ``source``. */
export type SwapTaggerBody = Omit<ActiveTaggerSelection, 'source'>;

/** Response from ``POST /api/tagger/swap`` — same shape as the GET on success. */
export type SwapTaggerResponse = ActiveTaggerResponse;

/** Sentinel id in the curated catalog that triggers the local-file prompt. */
export const LOCAL_FILE_ID = '__local__';

// --- Tag autocomplete (Tags tab custom-tag input) ---

/** One ranked tag from the suggestion endpoint. Mirrors the matcher's
 *  ``(name, category)`` tuple — reshaped to a dict at the API boundary for
 *  JSON-friendliness. ``category`` is the danbooru taxonomy
 *  (``general`` / ``artist`` / ``copyright`` / ``character`` / ``meta``). */
export interface TagSuggestion {
    name: string;
    category: string;
}

// --- Tag suggestion variant (autocomplete catalog selection) ---

/** One selectable catalog variant — ``value`` is the enum key sent back on
 *  PUT, ``label`` is the brand spelling shown in the dropdown. */
export interface SuggestionVariantOption {
    value: string;
    label: string;
}

/** ``GET`` / ``PUT /api/tagging/suggest/variant`` response. ``variant`` is the
 *  active selection (persisted user override wins over ``default``);
 *  ``default`` is the ``Configuration`` default; ``variants`` populates the
 *  picker. */
export interface SuggestionVariantResponse {
    variant: string;
    default: string;
    variants: SuggestionVariantOption[];
}
