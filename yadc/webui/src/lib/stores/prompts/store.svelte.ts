/**
 * Reactive state for the in-progress (or just-finished) generation.
 *
 * Uses Svelte 5 runes (``$state``) so components can read it
 * directly without a writable + subscribe dance. Single source of
 * truth for the streaming body, reasoning, status, and the active
 * ``AbortController`` (so ``cancelGeneration`` can abort it).
 *
 * Module-level (singleton) state — there is only one generation
 * at a time, and the page is single-instance.
 */

import type { GenerationStatus } from './types';

interface GenerationState {
    status: GenerationStatus;
    body: string;
    reasoning: string;
    error: string | null;
    controller: AbortController | null;
}

const _state = $state<GenerationState>({
    status: 'idle',
    body: '',
    reasoning: '',
    error: null,
    controller: null
});

/** Reactive state object — read in components (no binding required). */
export const generation: GenerationState = _state;

/** Append a token to the streamed body. */
export function appendToken(text: string): void {
    _state.body += text;
}

/** Append reasoning text. */
export function appendReasoning(text: string): void {
    _state.reasoning += text;
}

/** Start a generation — creates a fresh AbortController, clears prior state. */
export function beginGeneration(): AbortController {
    const controller = new AbortController();
    _state.status = 'streaming';
    _state.body = '';
    _state.reasoning = '';
    _state.error = null;
    _state.controller = controller;
    return controller;
}

/** Mark the generation as successfully complete. */
export function completeGeneration(): void {
    if (_state.status === 'streaming') {
        _state.status = 'done';
    }
    _state.controller = null;
}

/** Mark the generation as cancelled (user clicked Cancel). */
export function cancelGenerationState(): void {
    _state.controller?.abort();
    _state.status = 'cancelled';
    _state.controller = null;
}

/** Mark the generation as errored with a message. */
export function failGeneration(message: string): void {
    _state.status = 'error';
    _state.error = message;
    _state.controller = null;
}

/** Reset to idle (e.g. when the user starts a new generation). */
export function reset(): void {
    _state.status = 'idle';
    _state.body = '';
    _state.reasoning = '';
    _state.error = null;
    _state.controller = null;
}
