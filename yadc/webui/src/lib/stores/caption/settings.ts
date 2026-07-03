import storable from '$lib/storable.js';
import { z } from 'zod';

export interface CaptionSettings {
    $version: number;
    env: string;
    maxTokens: number;
    imageQuality: 'auto' | 'high' | 'low';
    draftName: string;
    overwrite: boolean;
    rounds: number;
    /** Number of in-flight `predict_stream` requests to allow at once. 1 = sequential. */
    batchSize: number;
    reasoningEnabled: boolean;
    reasoningEffort: 'low' | 'medium' | 'high';
    selectedTemplate: string;
    /** Last-used API url — used as a guard to decide whether to restore apiModelName. */
    apiUrl: string;
    /** Last-used model name — only restored when the current API url matches apiUrl. */
    apiModelName: string;
}

const CaptionSettingsSchema = z.object({
    $version: z.number(),
    env: z.string(),
    maxTokens: z.number(),
    imageQuality: z.enum(['auto', 'high', 'low']),
    draftName: z.string(),
    overwrite: z.boolean(),
    rounds: z.number(),
    batchSize: z.number(),
    reasoningEnabled: z.boolean(),
    reasoningEffort: z.enum(['low', 'medium', 'high']),
    selectedTemplate: z.string(),
    apiUrl: z.string(),
    apiModelName: z.string()
});

export const captionSettings = storable(
    'yadc/captionSettings',
    {
        $version: 2,
        env: 'default',
        maxTokens: 512,
        imageQuality: 'auto',
        draftName: '',
        overwrite: false,
        rounds: 1,
        batchSize: 1,
        reasoningEnabled: false,
        reasoningEffort: 'low',
        selectedTemplate: 'default',
        apiUrl: '',
        apiModelName: ''
    },
    null,
    CaptionSettingsSchema
);
