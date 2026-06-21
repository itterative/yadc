// --- Types matching the backend API ---

/** A few-shot example: image (already base64-encoded as a data URL) + subject + caption. */
export interface ExamplePair {
    subject: string;
    caption: string;
    image_data_url: string;
}

export type PromptGenFocus = 'system' | 'user' | 'both';

export type PromptImageQuality = 'auto' | 'low' | 'high';

export interface PromptGenRequest {
    env: string;
    intent: string;
    examples: ExamplePair[];
    focus: PromptGenFocus;
    api_model_name?: string | null;
    /** Output token cap for the generation call (client default of 4096
     *  truncates longer templates). Defaults to 16384 if omitted. */
    max_tokens?: number;
    /** Few-shot example image fidelity — maps to OpenAI ``image_url.detail``
     *  and Gemini ``mediaResolution``. Defaults to ``'auto'``. */
    image_quality?: PromptImageQuality;
    /** When set, runs in refine mode: the model applies the user's
     *  intent to this existing template instead of inventing a new
     *  one from scratch. ``null``/omitted → generate mode. */
    template_content?: string | null;
}

/** NDJSON streaming events from POST /api/prompts/generate. */
export type StreamEvent =
    | { type: 'token'; text?: string; reasoning?: string; reasoning_summary?: string }
    | { type: 'done' }
    | { type: 'error'; message: string };

/** Status of the in-progress (or just-finished) generation. */
export type GenerationStatus = 'idle' | 'streaming' | 'done' | 'cancelled' | 'error';

// --- Prompt history types (Phase 7) ---

/** A summary row in the history list (no full examples payload, no full intent). */
export interface PromptHistoryListItem {
    id: number;
    mode: 'generate' | 'refine';
    had_template: boolean;
    focus: PromptGenFocus;
    intent_preview: string;
    example_count: number;
    created_t: number;
}

/** A single prompt-history entry, fetched via ``GET /api/prompts/history/<id>`` for restore. */
export interface PromptHistoryEntry {
    id: number;
    mode: 'generate' | 'refine';
    intent: string;
    focus: PromptGenFocus;
    examples: ExamplePair[];
    template_content: string | null;
    created_t: number;
}

/** Body for ``POST /api/prompts/history`` (save current as a history entry). */
export interface SaveHistoryArgs {
    mode: 'generate' | 'refine';
    intent: string;
    focus: PromptGenFocus;
    examples: ExamplePair[];
    template_content: string | null;
}
