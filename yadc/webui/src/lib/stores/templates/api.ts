import { API_BASE, apiErrorMessage } from '$lib/api';
import { debounce } from '$lib/async';
import type { TemplateInfo, TemplateListItem } from './store';

async function _fetchTemplates(signal?: AbortSignal): Promise<TemplateListItem[]> {
    const res = await fetch(`${API_BASE}/api/templates`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced template list fetch — dedupes simultaneous component loads. */
export const fetchTemplates = debounce(_fetchTemplates);

async function _fetchTemplate(name: string, signal?: AbortSignal): Promise<TemplateInfo> {
    const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`, { signal });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Debounced single-template fetch — dedupes rapid selection changes. */
export const fetchTemplate = debounce(_fetchTemplate);

export async function saveTemplate(
    name: string,
    content: string,
    signal?: AbortSignal
): Promise<TemplateInfo> {
    const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
    return res.json();
}

/** Duplicate a template to a new name.
 *
 *  Loads the source template's content and saves it under ``newName``
 *  via the existing ``PUT /templates/<name>`` endpoint — no new API
 *  surface needed. The watcher picks up the new file and emits
 *  ``templates_changed`` via SSE, so the templates list auto-refreshes.
 *
 *  If ``newName`` already exists, the existing template is overwritten
 *  (matches ``PUT`` semantics). Callers that want to warn on collision
 *  should check ``existingNames`` up front.
 */
export async function duplicateTemplate(
    srcName: string,
    newName: string,
    signal?: AbortSignal
): Promise<TemplateInfo> {
    const source = await fetchTemplate(srcName, signal);
    return saveTemplate(newName, source.content, signal);
}

export async function deleteTemplate(name: string, signal?: AbortSignal): Promise<void> {
    const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res));
    }
}
