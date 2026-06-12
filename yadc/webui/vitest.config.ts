import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'jsdom',
        include: ['src/**/*.test.ts'],
        setupFiles: ['src/tests/setup.ts']
    },
    resolve: {
        alias: {
            // SvelteKit's $app/environment is not available in vitest.
            // The setup file mocks it to report `browser: true`.
            '$app/environment': new URL('./src/tests/mock-app-environment.ts', import.meta.url)
                .pathname,
            $lib: new URL('./src/lib', import.meta.url).pathname
        }
    }
});
