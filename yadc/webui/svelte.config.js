import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),

    kit: {
        adapter: adapter(),
        router: {
            type: 'hash'
        },
        alias: {
            // Points to the yadc Python package root, which is a sibling
            // of this webui/ directory. Lets Svelte components import
            // shared package resources (e.g. bundled Jinja templates)
            // without fragile deep `..` chains.
            $yadc: '../'
        }
    },

    onwarn: (warning, handler) => {
        if (warning.code.startsWith('a11y')) {
            return;
        }
        handler(warning);
    }
};

export default config;
