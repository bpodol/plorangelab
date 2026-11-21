# REST API contract

## Overview

The REST API manages **sessions, community boards, player ranges, and filter trees**. Equity calculation itself is exposed through the [WebSocket protocol](websocket.md), not a REST calculation endpoint.

This document describes routes implemented by `src/plorangelab/api/routers/sessions.py`, `players.py`, and `filters.py`, with schemas from `src/plorangelab/api/schemas.py`. It is a client-facing contract; see [backend API architecture](../backend/api/README.md) for synchronization and implementation details.

Paths below are relative to the API host. Bodies and responses use JSON except where no body is returned. `{session_id}` identifies an in-memory session and `{player_id}` is a **zero-based integer 0 through 5**.

## Endpoint index

| Method | Path | Success | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/sessions` | 201 | Create an empty session. |
| `PUT` | `/api/sessions/{session_id}/mode` | 204 | Change the session-wide PLO mode. |
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

The `/filters/{name}` and `/groups/{name}` aliases share handlers; the service identifies a node by **name and parent-group path**, not by the URL segment's asserted type. No session-list, session-detail, or session-delete REST route is present in the uploaded routers.

## Sessions and board

### Create session

`POST /api/sessions`

No request body is defined. Creates a session with six empty player slots, a preflop street state, no community board, and PLO4 as the default session-wide mode.

**201 response** (`SessionResponse`; illustrative identifier):

```json
{
  "session_id": "example-session-id"
}
```

A session ID is required for subsequent requests. A successful session lookup refreshes its activity; inactivity expiration is checked on lookup, with a **30-minute** timeout when no unfinished equity job exists.

### Change PLO mode

`PUT /api/sessions/{session_id}/mode`

**Request** (`ModeRequest`):

```json
{ "mode": 5 }
```

`mode` is required and must be `4` or `5`.

The mode is shared by all players and streets within the session. A successful change determines how subsequently added ranges are encoded and how equity calculations are performed.

Changing to the currently selected mode is a no-op, even when player ranges exist.

Changing to a different mode is rejected if any player range exists in the session. The mode change does not clear the board or existing street states.

**204 response:** no response body.

**409 response:** a different mode was requested while player ranges exist in the session, or an unfinished equity calculation prevents mutation.

**422 response:** invalid mode value.

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


**200 response** (`BoardResponse`; illustrative response after clearing the board to preflop):

```json
{
  "players": [
    {
      "base_hands": 1200,
      "filtered_hands": 1200,
      "filtered_coverage": 1.0,
      "filters": {}
    },
    null,
    null,
    null,
    null,
    null
  ]
}
```

`players` contains six current-street snapshots in player slot order. Empty slots contain `null`.

Each occupied slot includes:
- `base_hands`: Number of hands in the current street's base range.
- `filtered_hands`: Number of hands remaining after current-street filtering.
- `filtered_coverage`: Weighted coverage relative to the current street's base range.
- `filters`: Complete filter tree for the current street.

On a newly constructed flop, turn, or river state, the backend automatically generates a filter tree for each occupied player. Generated nodes are initially inactive, so generation alone does not reduce `filtered_hands`. The example above shows the preflop state, where automatic postflop filters do not apply.

Returning to a previously visited compatible street reuses its existing filter state without regeneration. Setting the board to `null` returns players to preflop and discards their postflop street states.

See [Range Management](../backend/services/range-management.md).


## Player ranges

### Create or replace a range

`PUT /api/sessions/{session_id}/players/{player_id}`

**Request** (`RangeRequest`):

```json
{ "hands": "50%" }
```

- `hands` is a required range-expression string. The encoder interprets supported percentage and explicit weighted-hand notation; see [encoding](../backend/encoding.md).
- The range is interpreted using the session's current PLO mode.
- `player_id` must be in **0 through 5**.

A new range replaces the selected player's range and filter configurations across all existing streets. The manager recreates that player's preflop range and propagates it through existing postflop boards, removing blocked hands. It generates automatic filters **only for the currently active street**, if that street has a board; inactive cached streets are not eagerly regenerated for the replaced player. Other player slots remain unchanged.

**200 response** (`PlayerFilterSnapshotResponse`; illustrative preflop example):

```json
{
  "base_hands": 2,
  "filtered_hands": 2,
  "filtered_coverage": 1.0,
  "filters": {}
}
```

The response has the same snapshot fields as board updates and filter mutations. On an active postflop board, `filters` contains the newly generated filter tree; its nodes are initially inactive. `base_hands` is the number of combinations after board blockers and any applicable parent-street range selection. `filtered_hands` and `filtered_coverage` describe the active street's effective selection.

### Remove a range

`DELETE /api/sessions/{session_id}/players/{player_id}`

**204 response:** no response body. Clears the selected player's slot across all existing street states while leaving the shared mode, boards, and other player slots unchanged. Removing an empty slot is allowed.

## Filters and groups

All filter routes operate on the **specified player's active street**. Saved filters represent evaluated membership selections; activation controls which of those selections currently affect that player's range.

### Shared mutation response

All successful filter and group mutations return a `PlayerFilterSnapshotResponse` containing the affected player's updated current-street state.

For example, creating an inactive root-level filter might return:

```json
{
  "base_hands": 4,
  "filtered_hands": 4,
  "filtered_coverage": 1.0,
  "filters": {
    "aces": {
      "type": "filter",
      "name": "aces",
      "expression": "AA",
      "coverage": 0.25,
      "active": false
    }
  }
}
```

Values are illustrative.

`filtered_coverage` represents the effective weighted coverage of the current street's base range. It is not necessarily relative to the original preflop range, because filtering from previous streets may already have narrowed the current street's base range.

Each filter or group's `coverage` describes its membership coverage independently of activation. Therefore, creating an inactive filter can leave `filtered_coverage` at `1.0`.

The response contains the complete resulting filter tree, not only the modified node.

### Addressing nested nodes

On **POST** creation requests, `path` is a JSON array of parent group names, starting at the root:

```json
{
  "expression": "As",
  "name": "ace_of_spades",
  "path": ["draws", "blockers"]
}
```

On **PATCH** and **DELETE** node routes, supply the same parent path as **repeated query parameters**:

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

**201 response** (`PlayerFilterSnapshotResponse`; illustrative):

```json
{
  "base_hands": 4,
  "filtered_hands": 4,
  "filtered_coverage": 1.0,
  "filters": {
    "aces": {
      "type": "filter",
      "name": "aces",
      "expression": "AA",
      "coverage": 0.25,
      "active": false
    }
  }
}
```

The new filter's `coverage` describes its weighted membership coverage relative to the current street's base range. The snapshot's `filtered_coverage` describes the player's effective range coverage.

A root-level filter is created inactive. A nonempty filter created directly inside an active group may start active. Zero-match filters may be saved but not directly activated.

### Create a group

`POST /api/sessions/{session_id}/players/{player_id}/groups`

**Request** (`GroupCreateRequest`):

```json
{ "name": "draws", "path": [] }
```

`name` is required and nonempty; `path` is an optional array of parent group names. New groups are always empty and inactive, even inside active groups.

**201 response** (`PlayerFilterSnapshotResponse`; illustrative):

```json
{
  "base_hands": 4,
  "filtered_hands": 4,
  "filtered_coverage": 1.0,
  "filters": {
    "draws": {
      "type": "group",
      "name": "draws",
      "coverage": 0.0,
      "active": false,
      "filters": {}
    }
  }
}
```

The response includes the newly created group in the complete filter tree. New groups are always empty and inactive, including groups created inside active parent groups.

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

**200 response** (`PlayerFilterSnapshotResponse`; illustrative):

```json
{
  "base_hands": 4,
  "filtered_hands": 1,
  "filtered_coverage": 0.25,
  "filters": {
    "aces": {
      "type": "filter",
      "name": "aces",
      "expression": "AA",
      "coverage": 0.25,
      "active": true
    }
  }
}
```

The response reflects the resulting activation state, filtered hand count, and effective weighted coverage.

The snapshot's `filtered_coverage` may differ from the addressed node's membership `coverage`.

### Clear filters

`POST /api/sessions/{session_id}/players/{player_id}/filters/clear`

No request body is used. **Deactivates** all filters and groups for that street while retaining definitions and hierarchy. The range reverts to that street's base range.

**200 response** (`PlayerFilterSnapshotResponse`; illustrative):

```json
{
  "base_hands": 4,
  "filtered_hands": 4,
  "filtered_coverage": 1.0,
  "filters": {
    "aces": {
      "type": "filter",
      "name": "aces",
      "expression": "AA",
      "coverage": 0.25,
      "active": false
    }
  }
}
```

The effective range returns to the street's base range, so `filtered_hands` equals `base_hands` and `filtered_coverage` becomes `1.0`.

### Delete a node

Either route:

```text
DELETE /api/sessions/{session_id}/players/{player_id}/filters/{name}
DELETE /api/sessions/{session_id}/players/{player_id}/groups/{name}
```

An optional repeated `path` query identifies the parent group hierarchy. Deleting a group deletes all descendants.

**200 response** (`PlayerFilterSnapshotResponse`; illustrative):

```json
{
  "base_hands": 4,
  "filtered_hands": 4,
  "filtered_coverage": 1.0,
  "filters": {}
}
```

The response contains the complete updated filter tree with the deleted node removed. Deleting a group also removes its descendants.

The effective hand count and weighted coverage are recalculated based on the remaining active filters. In this example, deleting the only filter restores the unfiltered base range.



## Error responses and validation

| Status | When |
| --- | --- |
| **404** | A session is unknown, expired, or closed. |
| **409** | A mutation is attempted while an equity calculation is unfinished, or a different PLO mode is requested while active-street player ranges exist. |
| **422** | Invalid request field/path value or invalid domain operation, such as a bad board, range, filter expression, group path, or missing mutation target. |

The API has **two 422 response styles**:

- Request validation errors generated by FastAPI/Pydantic typically contain a structured list under `detail`.
- Domain `ValueError` exceptions caught by routers produce a string under `detail`, for example `{"detail": "..."}`.

Missing filter or group nodes during mutations produce HTTP 422. The old filter-tree GET endpoints are no longer exposed; complete filter trees are returned through player-range PUT responses, board-change responses, and successful filter mutation responses. The code also does not provide an explicit HTTP `DELETE /api/sessions/{session_id}` route.

## Concurrency and session lifetime

REST mutations, including mode changes, run under `editable_session()`, which acquires the session lock and rejects unfinished equity jobs with HTTP 409. Service-level range and filter updates prepare replacement state before publishing successful changes, but clone-based failure atomicity is not a substitute for synchronization.

Retrieving a session through an API dependency refreshes its activity. Sessions can expire on lookup after **30 minutes** of inactivity when no job is unfinished; expiration is not performed by a periodic REST endpoint or background sweeper in the current implementation.

## Related documentation

- [Backend API architecture](../backend/api/README.md)
- [WebSocket protocol](websocket.md)
- [Range management](../backend/services/range-management.md)
- [Filter-expression engine](../backend/filter-engine.md)
- [Encoding](../backend/encoding.md)
