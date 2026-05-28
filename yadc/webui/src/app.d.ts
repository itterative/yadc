// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
interface ImportMetaEnv {
    readonly PUBLIC_API_DEFAULT_DEBOUNCE_MS: string;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface ImportMeta {
    readonly env: ImportMetaEnv;
}

declare global {
    namespace App {
        // interface Error {}
        // interface Locals {}
        // interface PageData {}
        // interface PageState {}
        // interface Platform {}
    }
}

export {};
