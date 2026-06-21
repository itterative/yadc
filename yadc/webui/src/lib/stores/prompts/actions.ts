/**
 * High-level operations for the prompt generator: ``startGeneration``
 * kicks off the streaming. ``cancelGeneration`` lives in the store
 * since it operates on the state-owned AbortController.
 *
 * ``startGeneration`` wires the streaming callbacks to the state
 * mutators and surfaces errors via the toasts store (for transient
 * feedback) and the generation state (for in-page error display).
 * The returned promise resolves once the stream ends (one way or
 * another) so the UI can re-enable its controls.
 */

import { generatePrompt } from './api';
import {
    appendReasoning,
    appendToken,
    beginGeneration,
    completeGeneration,
    failGeneration
} from './store.svelte';
import { toast } from '$lib/stores/toasts';
import { friendlyErrorMessage } from '$lib/api';
import type { ExamplePair, PromptGenFocus, PromptImageQuality, PromptGenRequest } from './types';

export interface StartGenerationArgs {
    env: string;
    intent: string;
    examples: ExamplePair[];
    focus: PromptGenFocus;
    apiModelName?: string | null;
    /** Output token cap for the generation call. */
    maxTokens?: number;
    /** Few-shot example image fidelity. */
    imageQuality?: PromptImageQuality;
    /** Existing template body for refine mode. ``null``/omitted → generate mode. */
    templateContent?: string | null;
}

/** Start a streaming generation. Resolves when the stream ends. */
export async function startGeneration(args: StartGenerationArgs): Promise<void> {
    const controller = beginGeneration();

    const request: PromptGenRequest = {
        env: args.env,
        intent: args.intent,
        examples: args.examples,
        focus: args.focus,
        api_model_name: args.apiModelName ?? null,
        max_tokens: args.maxTokens,
        image_quality: args.imageQuality,
        template_content: args.templateContent ?? null
    };

    try {
        await generatePrompt(
            request,
            {
                onToken: (text) => appendToken(text),
                onReasoning: (text) => appendReasoning(text),
                onDone: () => {
                    completeGeneration();
                },
                onError: (message) => {
                    // Mid-stream error — surface in the page state
                    // AND toast. The backend has already closed the
                    // stream, so nothing more to do here.
                    failGeneration(message);
                    toast.error(`Generation failed: ${message}`);
                }
            },
            controller.signal
        );
        // If ``generatePrompt`` returned without calling ``onDone``
        // (e.g. stream ended on EOF), make sure the state reflects
        // completion. Idempotent.
        completeGeneration();
    } catch (e) {
        if (controller.signal.aborted) {
            // User clicked Cancel — the store already reflects
            // 'cancelled' (set by ``cancelGeneration``). Don't toast.
            return;
        }
        const message = friendlyErrorMessage(e, 'Generation failed');
        failGeneration(message);
        toast.error(message);
    }
}
