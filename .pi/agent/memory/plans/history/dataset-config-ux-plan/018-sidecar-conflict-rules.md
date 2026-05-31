---
date: 2026-05-31
---
# Sidecar Conflict Rules for Append Uploads

## Decision

For **append uploads** to managed datasets, sidecars (`.toml`, `.txt`, `.draft~`) follow these rules:

### 1. Orphan Check — Expanded

A sidecar is accepted if **either**:
- Its image is in the **upload batch** (existing behavior), **OR**
- Its image already **exists in the live dataset** (new)

If neither → dropped as orphan.

This allows updating captions/extras for existing images without re-uploading the image itself.

### 2. Conflict List — Grouped by Stem

Conflicts are grouped by image stem. The conflict list shows:

| Scenario | Conflict list entry |
|----------|-------------------|
| Upload `foo.jpg` + `foo.toml` | `foo.jpg` (image-led group) |
| Upload `foo.toml` only, `foo.jpg` exists live | `foo.jpg` (sidecars for existing image) |
| Upload `foo.toml` only, `foo.jpg` NOT live | Dropped as orphan — no entry |

**Sidecars never appear as standalone entries** in the conflict list. They follow the resolution of their group.

### 3. Resolution Propagation

When a group is resolved:
- `overwrite` → overwrite image + all sidecars in the group
- `skip` → skip image + all sidecars in the group
- `keep_both` → rename image + all sidecars with `_1`, `_2` suffix

### 4. Sidecar Naming Convention

For image `foo.jpg`, the sidecar is `foo.toml` (not `foo.jpg.toml`).
