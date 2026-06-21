// --- Types matching the backend API ---

/** A few-shot example: image (already base64-encoded as a data URL) + subject + caption. */
export interface ExamplePair {
    subject: string;
    caption: string;
    image_data_url: string;
}

export type PromptGenFocus = 'system' | 'user' | 'both';

export interface PromptGenRequest {
    env: string;
    intent: string;
    examples: ExamplePair[];
    focus: PromptGenFocus;
    api_model_name?: string | null;
}

/** NDJSON streaming events from POST /api/prompts/generate. */
export type StreamEvent =
    | { type: 'token'; text?: string; reasoning?: string; reasoning_summary?: string }
    | { type: 'done' }
    | { type: 'error'; message: string };

/** Status of the in-progress (or just-finished) generation. */
export type GenerationStatus = 'idle' | 'streaming' | 'done' | 'cancelled' | 'error';
