# REST API contract

## Overview

The REST API manages **sessions, community boards, player ranges, and filter trees**. Equity calculation itself is exposed through the [WebSocket protocol](websocket.md), not a REST calculation endpoint.

This document describes routes implemented by `src/plorangelab/api/routers/sessions.py`, `players.py`, and `filters.py`, with schemas from `src/plorangelab/api/schemas.py`. It is a client-facing contract; see [backend API architecture](../backend/api/README.md) for synchronization and implementation details.

Paths below are relative to the API host. Bodies and responses use JSON except where no body is returned. `{session_id}` identifies an in-memory session and `{player_id}` is a **zero-based integer 0 through 5**.

## Endpoint index

| Method | Path | Success | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/sessions` | 201 | Create an empty session. |
| `PUT` | `/api/sessions/{session_id}/board` | 200 | Set or clear the shared board. |
| `PUT` | `/api/sessions/{session_id}/players/{player_id}` | 200 | Create or replace a player's range. |
| `DELETE` | `/api/sessions/{session_id}/players/{player_id}` | 204 | Clear a player slot. |
| `POST` | `/api/sessions/{session_id}/players/{player_id}/filters` | 201 | Save a filter. |
| `POST` | `/api/sessions/{session_id}/players/{player_id}/groups` | 201 | Create a group. |
| `PATCH` | `/api/sessions/{session_id}/players/{player_id}/filters/{name}` | 200 | Activate/deactivate a filter or group node. |
| `PATCH` | `/api/sessions/{session_id}/players/{player_id}/groups/{name}` | 200 | Same activation operation. |
| `POST` | `/api/sessions/{session_id}/players/{player_id}/filters/clear` | 200 | Deactivate all filters without deleting them. |
| `DELETE` | `/api/sessions/{session_id}/players/{player_id}/filters/{name}` | 200 | Delete a filter or group subtree. |
| `DELETE` | `/api/sessions/{session_id}/players/{player_id}/groups/{name}` | 200 | Same deletion operation. |
| `GET` | `/api/sessions/{session_id}/players/{player_id}/filters` | 200 | Retrieve the filter tree. |
| `GET` | `/api/sessions/{session_id}/players/{player_id}/filters/{name}` | 200 | Retrieve one node. |
| `GET` | `/api/sessions/{session_id}/players/{player_id}/groups/{name}` | 200 | Same node lookup. |

The `/filters/{name}` and `/groups/{name}` aliases share handlers; the service identifies a node by **name and parent-group path**, not by the URL segment's asserted type. No session-list, session-detail, or session-delete REST route is present in the uploaded routers.

## Sessions and board

### Create session

`POST /api/sessions`

No request body is defined. Creates a session with six empty player slots and no board.

**201 response** (`SessionResponse`; illustrative identifier):

```json
{
  "session_id": "example-session-id"
}
```

A session ID is required for subsequent requests. A successful session lookup refreshes its activity; inactivity expiration is checked on lookup, with a **30-minute** timeout when no unfinished equity job exists.

### Update or clear board

`PUT /api/sessions/{session_id}/board`

**Request** (`BoardRequest`):

```json
{ "board": "AsKd7h" }
```

`board` is **required** but may be `null` to clear the board:

```json
{ "board": null }
```

A non-null string must describe a valid flop (3 cards), turn (4 cards), or river (5 cards) in concatenated two-character card notation. Full card validation is performed by the underlying encoder/service, not the Pydantic field type.

**200 response** (`BoardResponse`; illustrative hand counts):

```json
{ "players": [1200, 0, 875, 0, 0, 620] }
```

`players` contains **six hand counts in slot order**, not player IDs. Empty slots have count `0`. Changing the board can invalidate later street filter states; setting `null` returns occupied players to preflop and discards postflop states. See [range management](../backend/services/range-management.md).

## Player ranges

### Create or replace a range

`PUT /api/sessions/{session_id}/players/{player_id}`

**Request** (`RangeRequest`):

```json
{ "hands": "50%", "mode": 4 }
```

- `hands` is a required range-expression string. The encoder interprets supported percentage and explicit weighted-hand notation; see [encoding](../backend/encoding.md).
- `mode` is required and must be **4** or **5**.
- `player_id` must be in **0–5**.

A new range **replaces** the slot's previous player state, including its saved filter configurations. If a board is already set, blocked hands are excluded before returning.

**200 response** (`RangeResponse`; illustrative count):

```json
{ "hands": 1248 }
```

The count is the player's **current** range size, after applying the board if applicable.

### Remove a range

`DELETE /api/sessions/{session_id}/players/{player_id}`

**204 response:** no response body. Removes the player's range, street states, and filters, while leaving the shared board and other player slots unchanged. Removing an empty slot is allowed.

## Filters and groups

All filter routes operate on the **specified player's active street**. Saved filters represent evaluated membership selections; activation controls which of those selections currently affect that player's range.

### Addressing nested nodes

On **POST** creation requests, `path` is a JSON array of parent group names, starting at the root:

```json
{
  "expression": "As",
  "name": "ace_of_spades",
  "path": ["draws", "blockers"]
}
```

On **GET**, **PATCH**, and **DELETE** node routes, supply the same parent path as **repeated query parameters**:

```text
?path=draws&path=blockers
```

The target's `{name}` is **not** included in `path`. Omit `path` for a root-level node. Names should be URL-encoded when embedded in paths. Each name is unique within its parent group; separate groups may reuse the same child name.

### Create a filter

`POST /api/sessions/{session_id}/players/{player_id}/filters`

**Request** (`FilterCreateRequest`):

| Field | Type | Behavior |
| --- | --- | --- |
| `expression` | string | Required, nonempty; parsed by the filter engine. |
| `name` | string or null | Optional; defaults to the expression when omitted/null. |
| `path` | array of strings | Optional; defaults to the root (`[]`). |

**201 response** (`FilterCoverageResponse`; illustrative):

```json
{ "coverage": 0.35 }
```

Here `coverage` is the **new filter's membership coverage**, measured using weights relative to the active street's base range; it is **not necessarily the player's effective range coverage**. A root-level filter is created inactive. A nonempty filter created directly inside an active group may start active. Zero-match filters may be saved but not directly activated.

### Create a group

`POST /api/sessions/{session_id}/players/{player_id}/groups`

**Request** (`GroupCreateRequest`):

```json
{ "name": "draws", "path": [] }
```

`name` is required and nonempty; `path` is an optional array of parent group names. New groups are always empty and inactive, even inside active groups.

**201 response** (`GroupResponse`):

```json
{ "name": "draws", "path": [] }
```

### Activate or deactivate a node

Either route:

```text
PATCH /api/sessions/{session_id}/players/{player_id}/filters/{name}
PATCH /api/sessions/{session_id}/players/{player_id}/groups/{name}
```

Optional `?path=...` identifies parent groups.

**Request** (`SetActiveRequest`):

```json
{ "active": true }
```

Activating a leaf also activates its ancestor groups without activating siblings. Activating a group recursively activates descendants; deactivating a group recursively deactivates them. Activation of a node whose membership matches zero hands is rejected.

**200 response** (`FilterCoverageResponse`; illustrative):

```json
{ "coverage": 0.42 }
```

Here `coverage` is the **resulting effective weighted coverage** of the player's active street, not the addressed node's total membership coverage.

### Clear filters

`POST /api/sessions/{session_id}/players/{player_id}/filters/clear`

No request body is used. **Deactivates** all filters and groups for that street while retaining definitions and hierarchy. The range reverts to that street's base range.

**200 response:**

```json
{ "coverage": 1.0 }
```

### Delete a node

Either route:

```text
DELETE /api/sessions/{session_id}/players/{player_id}/filters/{name}
DELETE /api/sessions/{session_id}/players/{player_id}/groups/{name}
```

An optional repeated `path` query identifies the parent group hierarchy. Deleting a group deletes all descendants. **200 response**: `FilterCoverageResponse` with the **updated effective coverage**.

### Retrieve the tree or a node

Full tree:

```text
GET /api/sessions/{session_id}/players/{player_id}/filters
```

Single node (both aliases are supported):

```text
GET /api/sessions/{session_id}/players/{player_id}/filters/{name}
GET /api/sessions/{session_id}/players/{player_id}/groups/{name}
```

Single-node lookup accepts repeated `path` query parameters.

**200 response**: a dictionary keyed by root node names (full tree), or one node object (single lookup). A leaf uses `FilterInfoResponse`; a group uses recursive `FilterGroupInfoResponse`:

```json
{
  "type": "group",
  "name": "draws",
  "coverage": 0.6,
  "active": true,
  "filters": {
    "flush_draw": {
      "type": "filter",
      "name": "flush_draw",
      "expression": "s",
      "coverage": 0.35,
      "active": true
    }
  }
}
```

*Illustrative shape and values only; examples are not claims about the matching behavior of these particular expressions.* For a full-tree response, this group object would be nested under the root key `"draws"`.

The `type` field is `"filter"` or `"group"`; group `filters` may contain nested groups and leaf filters. Node `coverage` is **membership coverage**, even for inactive nodes. The tree does not expose internal Boolean selection arrays or raw weighted hands.

## Error responses and validation

| Status | When |
| --- | --- |
| **404** | A session is unknown, expired, or closed; filter read endpoints also use 404 for missing players/nodes. |
| **409** | A mutation is attempted while a session's equity calculation has an unfinished future. |
| **422** | Invalid request field/path value; invalid domain operation (bad board/range/filter/group path, missing mutation target, duplicate name, etc.). |

The API has **two 422 response styles**:

- Request validation errors generated by FastAPI/Pydantic typically contain a structured list under `detail`.
- Domain `ValueError` exceptions caught by routers produce a string under `detail`, for example `{"detail": "..."}`.

A service-level missing node yields **422 for a mutation** but **404 for a read** in the current router implementation. The code also does not provide an explicit HTTP `DELETE /api/sessions/{session_id}` route.

## Concurrency and session lifetime

REST mutations run under `editable_session()`, which acquires the session lock and rejects any unfinished equity job with **409**. Filter reads use `locked_session()`, which checks that the session is open but does not reject reads merely because a calculation is running.

Retrieving a session through an API dependency refreshes its activity. Sessions can expire on lookup after **30 minutes** of inactivity when no job is unfinished; expiration is not performed by a periodic REST endpoint or background sweeper in the reviewed code.

## Related documentation

- [Backend API architecture](../backend/api/README.md)
- [WebSocket protocol](websocket.md)
- [Range management](../backend/services/range-management.md)
- [Filter-expression engine](../backend/filter-engine.md)
- [Encoding](../backend/encoding.md)
