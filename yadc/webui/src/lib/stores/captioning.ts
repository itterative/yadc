import { writable } from "svelte/store";
import { z } from "zod";

// --- SSE event schemas ---

export const CaptioningStatusZ = z.object({
  status: z.enum(["idle", "running", "stopping", "error", "done"]),
  dataset_name: z.string(),
  processed: z.number(),
  total: z.number(),
  errors: z.number(),
  error: z.string().nullable(),
});

export type CaptioningStatus = z.infer<typeof CaptioningStatusZ>;

export const captioningStatus = writable<CaptioningStatus>({
  status: "idle",
  dataset_name: "",
  processed: 0,
  total: 0,
  errors: 0,
  error: null,
});

// --- Zod schemas for ping events ---

export const PingEventZ = z.object({
  time: z.string(),
});
