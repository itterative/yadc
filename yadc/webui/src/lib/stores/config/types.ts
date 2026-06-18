// --- Types matching the backend API ---

export interface ExportBackend {
    name: string;
    description: string;
    formats: string[];
    zip_only?: boolean;
}

export interface ExportResult {
    status: string;
    count: number;
    dataset: string;
    backend: string;
    format: string;
    source: string;
    output: string;
}

export interface ExportZipOptions {
    dataset: string;
    backend?: string;
    format?: string;
    source?: string;
    draft?: string;
    with_drafts?: string[];
    caption_extension?: string;
    include_images?: boolean;
}

export interface DatasetConfig {
    name: string;
    config_path: string;
}

/** Mirrors the Pydantic Config model in yadc/core/config.py. Keep in sync. */
export interface Config {
    api?: ConfigApi;
    prompt?: ConfigPrompt;
    settings?: ConfigSettings;
    reasoning?: ConfigReasoning;
    dataset?: ConfigDatasetEntry[];
    env?: string;
    interactive?: boolean;
    rounds?: number;
    caption_suffix?: string;
    overwrite_captions?: boolean;
}

export interface ConfigApi {
    url?: string;
    token?: string;
    model_name?: string;
}

export interface ConfigPrompt {
    name?: string;
    template?: string;
}

export interface ConfigSettings {
    max_tokens?: number;
    store_conversation?: boolean;
    image_quality?: 'auto' | 'high' | 'low';
    advanced?: ConfigSettingsAdvanced;
}

export interface ConfigSettingsAdvanced {
    system_role?: string;
    user_role?: string;
    assistant_role?: string;
    assistant_prefill?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
}

export interface ConfigReasoning {
    enable?: boolean;
    thinking_effort?: 'low' | 'medium' | 'high';
    exclude_from_output?: boolean;
    advanced?: ConfigReasoningAdvanced;
}

export interface ConfigReasoningAdvanced {
    thinking_start?: string;
    thinking_end?: string;
}

export interface ConfigDatasetEntry {
    path?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    images?: any[];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    extras?: Record<string, any>;
}

export interface ConfigValidationError {
    loc: string[];
    msg: string;
    type: string;
}

export interface DatasetConfigDetail {
    name: string;
    config_path: string;
    content: string;
    parsed: Config;
    validation_error?: ConfigValidationError[];
}

export interface ConfigHistoryEntry {
    id: number;
    dataset_name: string;
    content: string;
    created_t: number;
}

/** Paginated page of config history entries.
 *  ``next`` is an opaque cursor to pass back for the next (older)
 *  page; ``null`` means this is the last page. The encoding is an
 *  implementation detail of the server. */
export interface ConfigHistoryPage {
    entries: ConfigHistoryEntry[];
    next_token: string | null;
}
