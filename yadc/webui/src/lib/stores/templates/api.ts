import { API_BASE, apiErrorMessage } from '$lib/api';
import { debounce } from '$lib/async';
import type { TemplateInfo, TemplateListItem } from './store';

async function _fetchTemplates(): Promise<TemplateListItem[]> {
    const res = await fetch(`${API_BASE}/api/templates`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced template list fetch — dedupes simultaneous component loads. */
export const fetchTemplates = debounce(_fetchTemplates);

async function _fetchTemplate(name: string): Promise<TemplateInfo> {
    const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`);
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced single-template fetch — dedupes rapid selection changes. */
export const fetchTemplate = debounce(_fetchTemplate);

export async function saveTemplate(name: string, content: string): Promise<TemplateInfo> {
    const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

export async function deleteTemplate(name: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`, {
        method: 'DELETE'
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}
