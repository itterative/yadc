---
name: thin-events-vs-fat-events
description: Research thin events (notify-then-fetch) vs event-carried state transfer (fat events) for SSE patterns
---

# Thin Events vs Fat Events

**Context:** The YADC project currently uses event-carried state transfer (fat events) for SSE — e.g., `CaptioningStatusEvent` carries full job state, `ImageCaptionedEvent` carries image metadata + caption text. The notify-then-fetch pattern (thin events) was adopted for `EnvironmentsChangedEvent` and `DatasetChangedEvent`.

**Question:** Should the project standardize on one pattern? If so, which one? What are the real-world tradeoffs at YADC's scale?

## Fat Events (current for most events)
- Event carries the full data the client needs
- No second HTTP round-trip
- Couples event schema to API response shape
- Harder to evolve independently
- Works well when data is small and changes are infrequent

## Thin Events (current for `DatasetChangedEvent`, `EnvironmentsChangedEvent`)
- Event just says "something changed"
- Client fetches fresh data on demand
- Decoupled event schema from API
- Client can debounce/batch/skip
- One extra HTTP request per notification

## Things to research
- [ ] Are there events in YADC where the payload is large enough that thin events would noticeably reduce SSE bandwidth?
- [ ] Are there events where the extra fetch latency matters (e.g., real-time captioning progress)?
- [ ] Could a hybrid approach work: thin events by default, with optional inline data for small/common cases?
- [ ] How do other projects (e.g., Supabase realtime, PocketBase, etc.) handle this?

## Decision needed
Standardize on one pattern, document the convention, and refactor existing events if needed.

## Current Status (2026-05-28)

The project now uses a **hybrid approach**:

- **Fat events** for data that changes per-item and is consumed immediately (e.g. `ImageCaptionedEvent` carries caption text, `CaptioningStatusEvent` carries full job state)
- **Enriched thin events** for collection-level changes (e.g. `EnvironmentsChangedEvent` carries `envs: list[str]`, `TemplatesChangedEvent` carries `templates: list[str]`). Frontend still fetches the full list on receipt, but the event payload provides debugging context and future flexibility for targeted updates.
- **Pure thin events** for cases where no useful context can be provided (e.g. `DatasetChangedEvent` — the changed files are internal implementation details, frontend just refreshes)

The convention: **prefer enriched thin events** for collection changes, **fat events** for per-item data that avoids a round-trip.

## Notable exception: `image_tagged` in the Tags tab (2026-07-04)

The Tags tab (`components/dataset/detail/Tags.svelte`) now handles
`image_tagged` **thin-style**: the handler ignores the fat payload and
bumps a local nonce that re-triggers the cached-result fetch
(`fetchCachedTagResult`). Reason: the always-add / banned policy is a
backend read-time transform, and the SSE payload carries the result
computed with the *tagging request's* policy — which may be stale
relative to the *viewer's* current policy. Refetching applies the
viewer's current settings. This is the first per-item event handled
thin-style; if the pattern holds up it may inform the broader
standardization decision above.
