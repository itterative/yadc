import { API_BASE } from "$lib/api";

// --- Types matching the backend API ---

export interface EnvInfo {
  name: string;
  api_url: string | null;
  api_token: string | null; // masked as [REDACTED]
  api_model_name: string | null;
}

export interface EnvListResult {
  models: string[];
  default?: string;
}

// --- API helpers ---

export async function fetchEnvs(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/envs`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchEnv(name: string): Promise<EnvInfo> {
  const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function saveEnv(
  name: string,
  data: { api_url?: string; api_token?: string; api_model_name?: string },
): Promise<EnvInfo> {
  const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function deleteEnv(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
}

export async function fetchModels(name: string): Promise<EnvListResult> {
  const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(name)}/models`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}
