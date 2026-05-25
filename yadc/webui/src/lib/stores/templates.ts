import { API_BASE, apiErrorMessage } from '$lib/api';
import { writable, readonly, type Readable } from 'svelte/store';

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

// --- API helpers ---

export async function fetchTemplates(): Promise<TemplateListItem[]> {
	const res = await fetch(`${API_BASE}/api/templates`);
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}

export async function fetchTemplate(name: string): Promise<TemplateInfo> {
	const res = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(name)}`);
	if (!res.ok) {
		throw new Error(await apiErrorMessage(res));
	}
	return res.json();
}

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

// --- Jinja2 variable extraction (matches backend _extract_variables) ---

const JINJA_VAR_RE = /\{\{-?\s*(\w+)(?:\.[\w.]+)*\s*(?:\|[^}]*)?-?\}\}/g;
const JINJA_FOR_RE = /\{%[-\s]+for\s+\w+\s+in\s+(\w+)/g;
const JINJA_BUILTINS = new Set([
	'true',
	'false',
	'none',
	'True',
	'False',
	'None',
	'range',
	'lipsum',
	'dict',
	'namespace'
]);

export function extractVariables(template: string): string[] {
	const names = new Set<string>();

	let m: RegExpExecArray | null;
	JINJA_VAR_RE.lastIndex = 0;
	while ((m = JINJA_VAR_RE.exec(template)) !== null) {
		names.add(m[1]);
	}

	JINJA_FOR_RE.lastIndex = 0;
	while ((m = JINJA_FOR_RE.exec(template)) !== null) {
		names.add(m[1]);
	}

	for (const b of JINJA_BUILTINS) {
		names.delete(b);
	}

	return [...names].sort();
}
