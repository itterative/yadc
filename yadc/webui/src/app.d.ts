/// <reference types="@sveltejs/kit" />

declare namespace App {
  // interface Error {}
  // interface Locals {}
  // interface PageData {}
  // interface PageState {}
  // interface Platform {}
}

declare module "$env/dynamic/public" {
  export const PUBLIC_BACKEND_URL: string;
}
