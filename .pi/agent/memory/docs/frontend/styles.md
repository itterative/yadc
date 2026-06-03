---
name: frontend/styles
description: Frontend lib/styles/ — Tailwind @layer components files (badges, buttons, forms, overlays, utilities, animations).
category: architecture
---

# Frontend: `lib/styles/`

Tailwind `@layer components` files. The rules for what to put here (vs. inline utilities vs. Svelte `<style>`) are in `frontend-architecture` under "Styling Patterns".

```
yadc/webui/src/lib/styles/
  badges.css         # .badge
  buttons.css        # .btn variants
  forms.css          # .input
  overlays.css       # .dialog-panel, .alert-{error,warning,success,info}
  utilities.css      # .btn-bar
  animations.css     # spin keyframes etc.
```

**Cross-references:**
- Styling rules (when to use @layer vs inline vs Svelte style): `frontend-architecture` → Styling Patterns
- Tailwind v4 setup, theme colors, vite plugin order: `webui-frontend`
