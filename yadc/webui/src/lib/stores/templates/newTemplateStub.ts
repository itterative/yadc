/// <reference types="vite/client" />

import defaultJinja from '$yadc/templates/jinja/default.jinja?raw';

/** Stub content for a brand new user template.
 *
 * Matches the CLI's `yadc templates edit <name>` behavior, which
 * pre-fills the editor with `# Edit the template below` followed by
 * the built-in default template. Sharing the source via a `?raw`
 * import keeps the CLI and webui in lockstep — the file content is
 * inlined into the JS bundle at build time, so there is no runtime
 * fetch and no symlink fragility. */
export const NEW_TEMPLATE_STUB = `# Edit the template below\n\n${defaultJinja}`;
