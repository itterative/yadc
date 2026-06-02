import { API_BASE } from "$lib/api";
import { writable } from "svelte/store";

// --- Types matching the backend API dataclasses ---

export interface DatasetInfo {
  name: string;
  config_path: string | null;
  image_count: number;
  has_caption: number;
  has_toml: number;
  last_scanned_t: number | null;
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
}

export interface ImagePage {
  images: ImageInfo[];
  next_token: string | null;
}

export interface CaptionData {
  caption: string;
  extras: Record<string, unknown>;
  extras_raw?: string;
  drafts: Record<string, string>;
}

// --- API helpers ---

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE}/api/datasets`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchImages(
  datasetName: string,
  options: { limit?: number; afterId?: number } = {},
): Promise<ImagePage> {
  const params = new URLSearchParams();
  if (options.limit) params.set("limit", String(options.limit));
  if (options.afterId !== undefined) params.set("after_id", String(options.afterId));

  const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images?${params}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchCaption(datasetName: string, imageId: number): Promise<CaptionData> {
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`,
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function updateCaption(datasetName: string, imageId: number, caption: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/caption`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ caption }),
    },
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
}

export async function updateExtras(datasetName: string, imageId: number, extrasRaw: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/extras`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ extras_raw: extrasRaw }),
    },
  );
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.error) message = body.error;
    } catch { /* ignore */ }
    throw new Error(message);
  }
}

export function thumbnailUrl(datasetName: string, imageId: number, size = 256): string {
  return `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/thumbnail?size=${size}`;
}

export interface PromptPreview {
  system_prompt: string;
  user_prompt: string;
  template_context: Record<string, unknown>;
}

export async function fetchPromptPreview(
  datasetName: string,
  imageId: number,
  options: { template?: string; template_name?: string } = {},
): Promise<PromptPreview> {
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/preview-prompt`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    },
  );
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.error) message = body.error;
    } catch { /* ignore JSON parse failure */ }
    throw new Error(message);
  }
  return res.json();
}

export function mediaUrl(datasetName: string, imageId: number): string {
  return `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/images/${imageId}/media`;
}

// --- Captioning API helpers ---

export interface CaptioningJobInfo {
  status: "idle" | "running" | "stopping" | "error" | "done";
  dataset_name: string;
  processed: number;
  total: number;
  errors: number;
  error: string | null;
}

/** Start a captioning job. Returns initial job info. */
export async function startCaptioning(
  datasetName: string,
  options: Record<string, unknown>,
): Promise<CaptioningJobInfo> {
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    },
  );
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.error) {
        if (typeof body.error === "string") {
          message = body.error;
        } else if (Array.isArray(body.error)) {
          // Pydantic validation errors
          message = body.error.map((e: { msg: string }) => e.msg).join(", ");
        }
      }
    } catch { /* ignore */ }
    throw new Error(message);
  }
  return res.json();
}

/** Stop a running captioning job. */
export async function stopCaptioning(datasetName: string): Promise<boolean> {
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(datasetName)}/caption`,
    { method: "DELETE" },
  );
  if (!res.ok) return false;
  return true;
}

// --- Store for paginated image browsing ---

export interface DatasetBrowserState {
  images: ImageInfo[];
  isLoading: boolean;
  isLoadingMore: boolean;
  error: string | null;
  hasMore: boolean;
}

export function createDatasetBrowserStore(datasetName: string, pageSize = 50) {
  const { subscribe, set, update } = writable<DatasetBrowserState>({
    images: [],
    isLoading: false,
    isLoadingMore: false,
    error: null,
    hasMore: true,
  });

  let lastAfterId = 0;

  async function loadInitial() {
    lastAfterId = 0;
    update((s) => ({ ...s, isLoading: true, error: null }));

    try {
      const page = await fetchImages(datasetName, { limit: pageSize, afterId: 0 });

      if (page.images.length > 0) {
        lastAfterId = page.images[page.images.length - 1].id;
      }

      set({
        images: page.images,
        isLoading: false,
        isLoadingMore: false,
        error: null,
        hasMore: page.next_token !== null,
      });
    } catch (e) {
      update((s) => ({
        ...s,
        isLoading: false,
        error: e instanceof Error ? e.message : "Failed to load images",
      }));
    }
  }

  async function loadMore() {
    let currentState: DatasetBrowserState | undefined;
    update((s) => {
      currentState = s;
      return { ...s, isLoadingMore: true };
    });

    if (!currentState || currentState.isLoadingMore || !currentState.hasMore) return;

    try {
      const page = await fetchImages(datasetName, { limit: pageSize, afterId: lastAfterId });

      if (page.images.length > 0) {
        lastAfterId = page.images[page.images.length - 1].id;
      }

      update((s) => ({
        ...s,
        images: [...s.images, ...page.images],
        isLoadingMore: false,
        hasMore: page.next_token !== null,
      }));
    } catch (e) {
      update((s) => ({
        ...s,
        isLoadingMore: false,
        error: e instanceof Error ? e.message : "Failed to load more images",
      }));
    }
  }

  function updateImage(imageId: number, patch: Partial<ImageInfo>) {
    update((s) => ({
      ...s,
      images: s.images.map((img) => (img.id === imageId ? { ...img, ...patch } : img)),
    }));
  }

  return {
    subscribe,
    loadInitial,
    loadMore,
    updateImage,
  };
}
