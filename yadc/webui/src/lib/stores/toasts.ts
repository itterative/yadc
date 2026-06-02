/**
 * Centralized toast notification store.
 *
 * Provides a reactive list of active toasts and convenience helpers for
 * creating them. Toasts are simple text messages with a variant (info,
 * success, warning, error), optional action button, and auto-dismiss
 * duration.
 *
 * Usage:
 *   import { toast } from "$lib/stores/toasts";
 *   toast.success("Exported 42 images → /output");
 *   toast.error("Failed to start captioning", { duration: 0 });
 */

import { writable, readonly, type Readable } from "svelte/store";
// --- Types ---

export type ToastVariant = "info" | "success" | "warning" | "error";

export interface ToastAction {
  label: string;
  handler: () => void;
}

export interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
  /** ms until auto-dismiss. 0 = persistent (must be dismissed manually). */
  duration: number;
  /** @deprecated Use actions instead. */
  action?: ToastAction;
  actions?: ToastAction[];
}

export interface ToastOptions {
  message: string;
  variant?: ToastVariant;
  /** Override auto-dismiss duration (ms). 0 = persistent. */
  duration?: number;
  /** @deprecated Use actions instead. */
  action?: ToastAction;
  actions?: ToastAction[];
}

// --- Defaults ---

const DEFAULT_DURATIONS: Record<ToastVariant, number> = {
  info: 5000,
  success: 4000,
  warning: 8000,
  error: 0, // persistent — user must dismiss
};

const MAX_TOASTS = 5;

// --- Store ---

const _toasts = writable<Toast[]>([]);

/** Reactive list of active toasts. */
export const toasts: Readable<Toast[]> = readonly(_toasts);

// --- Actions ---

/** Add a toast and return its id. */
export function addToast(options: ToastOptions): string {
  const variant = options.variant ?? "info";
  // NOTE: crypto.randomUUID requires https
  const id = crypto.randomUUID?.() ?? Math.random().toString(36).slice(2, 10);
  const toast: Toast = {
    id,
    message: options.message,
    variant,
    duration: options.duration ?? DEFAULT_DURATIONS[variant],
    action: options.action,
    actions: options.actions,
  };

  _toasts.update((list) => {
    const next = [...list, toast];
    // Evict oldest if over capacity (only auto-dismissable ones)
    if (next.length > MAX_TOASTS) {
      const removable = next.findIndex((t) => t.duration > 0);
      if (removable !== -1) {
        next.splice(removable, 1);
      } else {
        next.shift();
      }
    }
    return next;
  });

  return id;
}

/** Dismiss a toast by id. */
export function dismissToast(id: string): void {
  _toasts.update((list) => list.filter((t) => t.id !== id));
}

/** Dismiss all active toasts. */
export function dismissAllToasts(): void {
  _toasts.set([]);
}

// --- Convenience helpers ---

export const toast = {
  info(message: string, opts?: Omit<ToastOptions, "message" | "variant">) {
    return addToast({ ...opts, message, variant: "info" });
  },
  success(message: string, opts?: Omit<ToastOptions, "message" | "variant">) {
    return addToast({ ...opts, message, variant: "success" });
  },
  warning(message: string, opts?: Omit<ToastOptions, "message" | "variant">) {
    return addToast({ ...opts, message, variant: "warning" });
  },
  error(message: string, opts?: Omit<ToastOptions, "message" | "variant">) {
    return addToast({ ...opts, message, variant: "error" });
  },
};
