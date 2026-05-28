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
