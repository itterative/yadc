import storable from '$lib/storable.js';

export interface CaptionSettings {
	$version: number;
	env: string;
	maxTokens: number;
	imageQuality: 'auto' | 'high' | 'low';
	draftName: string;
	overwrite: boolean;
	rounds: number;
	reasoningEnabled: boolean;
	reasoningEffort: 'low' | 'medium' | 'high';
	selectedTemplate: string;
	/** Last-used API url — used as a guard to decide whether to restore apiModelName. */
	apiUrl: string;
	/** Last-used model name — only restored when the current API url matches apiUrl. */
	apiModelName: string;
}

export const captionSettings = storable<CaptionSettings>('yadc/captionSettings', {
	$version: 1,
	env: 'default',
	maxTokens: 512,
	imageQuality: 'auto',
	draftName: '',
	overwrite: false,
	rounds: 1,
	reasoningEnabled: false,
	reasoningEffort: 'low',
	selectedTemplate: 'default',
	apiUrl: '',
	apiModelName: ''
});
