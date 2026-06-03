import { writable, readonly, type Readable } from 'svelte/store';
import { fetchTemplates } from './api';

// --- Types matching the backend API ---

export interface TemplateInfo {
    name: string;
    source: 'user' | 'builtin';
    content: string;
    variables: string[];
}

export interface TemplateListItem {
    name: string;
    source: 'user' | 'builtin';
}

// --- Reactive store ---

export interface TemplateStoreState {
    loaded: boolean;
    items: TemplateListItem[];
}

const _templates = writable<TemplateStoreState>({ loaded: false, items: [] });

/** Reactive store for available templates (user + built-in). */
export const templates: Readable<TemplateStoreState> = readonly(_templates);

/** Fetch the template list from the API and update the store. */
export async function refreshTemplates(): Promise<TemplateListItem[]> {
    const list = await fetchTemplates();
    _templates.set({ loaded: true, items: list });
    return list;
}
