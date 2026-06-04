// --- Types matching the backend API dataclasses ---

export interface DatasetInfo {
    name: string;
    source: 'upload' | 'import' | 'create';
    config_path: string | null;
    image_count: number;
    has_caption: number;
    has_toml: number;
    last_scanned_t: number | null;
    first_image_id: number | null;
}

export interface ImageInfo {
    id: number;
    file_name: string;
    path: string;
    has_caption: boolean;
    has_toml: boolean;
    width: number;
    height: number;
    draft_names: string[];
    last_modified_t: number | null;
    caption_error?: string;
    flash?: number;
    delete_path?: string;
}

export interface ImagePage {
    images: ImageInfo[];
    next_token: string | null;
}

export interface DatasetUploadResult {
    dataset: DatasetInfo;
    warnings: string[];
}

export interface UploadConflict {
    file: string;
    existing_size: number;
    new_size: number;
}

export interface UploadProgressEvent {
    phase: 'validating' | 'writing' | 'conflicts' | 'complete' | 'error';
    file?: string;
    index?: number;
    total?: number;
    dataset?: DatasetInfo;
    warnings?: string[];
    message?: string;
    staging_id?: string;
    conflicts?: UploadConflict[];
}

export interface CaptionData {
    caption: string;
    extras: Record<string, unknown>;
    extras_raw?: string;
    drafts: Record<string, string>;
}

export interface HistoryEntry {
    index: number;
    caption: string;
    extras: Record<string, unknown>;
    hash: string;
}

export interface DatasetFolder {
    name: string;
    path: string;
    image_count: number;
    can_delete: boolean;
}

export interface DraftSummary {
    name: string;
    image_count: number;
}

export interface PromptPreview {
    system_prompt: string;
    user_prompt: string;
    template_context: Record<string, unknown>;
    template_context_toml: string;
}

export interface CaptioningJobInfo {
    status: 'idle' | 'running' | 'stopping' | 'error' | 'done' | 'cancelled';
    dataset_name: string;
    job_id: string;
    processed: number;
    total: number;
    errors: number;
    error: string | null;
    error_messages: string[];
    api_url?: string;
    api_model_name?: string;
}
