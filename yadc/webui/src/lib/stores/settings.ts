import storable from "$lib/storable.js";
import { writable } from "svelte/store";

export interface Settings {
  $version: number;
  thumbnailsPerRow: number;
}

export const settings = storable<Settings>("yadc/settings", {
  $version: 1,
  thumbnailsPerRow: 5,
});

export const settingsOpen = writable<boolean>(false);
