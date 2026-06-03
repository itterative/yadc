// --- Jinja2 variable extraction (matches backend _extract_variables) ---

const JINJA_VAR_RE = /\{\{-?\s*(\w+)(?:\.[\w.]+)*\s*(?:\|[^}]*)?-?\}\}/g;
const JINJA_FOR_RE = /\{%[-\s]+for\s+\w+\s+in\s+(\w+)/g;
const JINJA_BUILTINS = new Set([
    'true',
    'false',
    'none',
    'True',
    'False',
    'None',
    'range',
    'lipsum',
    'dict',
    'namespace'
]);

export function extractVariables(template: string): string[] {
    const names = new Set<string>();

    let m: RegExpExecArray | null;
    JINJA_VAR_RE.lastIndex = 0;
    while ((m = JINJA_VAR_RE.exec(template)) !== null) {
        names.add(m[1]);
    }

    JINJA_FOR_RE.lastIndex = 0;
    while ((m = JINJA_FOR_RE.exec(template)) !== null) {
        names.add(m[1]);
    }

    for (const b of JINJA_BUILTINS) {
        names.delete(b);
    }

    return [...names].sort();
}
