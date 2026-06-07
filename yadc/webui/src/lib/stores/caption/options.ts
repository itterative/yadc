/**
 * Type for the options object assembled by CaptionSettings
 * and passed to the onstart callback. Mirrors CaptionJobOptions
 * from the backend, with undefined fields omitted.
 */
export interface CaptionOptions {
    env?: string;
    api_url?: string;
    api_token?: string;
    api_model_name?: string;
    prompt_template?: string;
    prompt_name?: string;
    max_tokens?: number;
    image_quality?: 'auto' | 'high' | 'low';
    rounds?: number;
    draft?: string;
    overwrite?: boolean;
    /** Number of in-flight `predict_stream` requests. Omitted when 1 (default). */
    max_concurrent?: number;
    reasoning?: boolean;
    reasoning_effort?: 'low' | 'medium' | 'high';
    password?: string;
}
