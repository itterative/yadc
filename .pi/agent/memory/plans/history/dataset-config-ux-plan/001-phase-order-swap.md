---
date: 2025-05-30
---
# Implementation Order Swap: Phase 4 Before Phase 1

**Context:** Phase 1 (dataset entries + extras) will significantly grow `DatasetConfig.svelte`. Phase 4 (shared components) extracts existing fields from that same file. Doing Phase 4 first avoids rework — extract first, then add onto the clean structure.

**Decision:** Implement Phase 4 (shared field components) before Phase 1 (dataset entries). The plan's formal phase numbering stays the same, but implementation order is: Phase 4 → Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6.

**Rationale:** Pure refactor first reduces risk. Shared component is quick to verify visually before building new features on top.
