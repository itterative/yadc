---
name: backend/templates
description: Built-in Jinja2 prompt templates under yadc/templates/ — the default template loaded via importlib.resources.
category: architecture
---

# Backend: Built-in Templates (`yadc/templates/`)

```
yadc/templates/
  __init__.py     # re-exports default_template, load_builtin_template
  templates.py    # loads from yadc/templates/jinja/ via importlib.resources
  jinja/
    default.jinja # built-in prompt template (system_prompt, user_prompt, user_prompt_multiple_rounds)
```

**Cross-references:**
- Template resolution chain (built-in → user template → inline string): `template-system`
