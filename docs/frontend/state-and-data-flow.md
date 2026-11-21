# Frontend state and data flow

## Overview

PLO RangeLab's React frontend coordinates session setup, player-range editing, community board changes, and live equity calculations. The current implementation separates **transport functions** (`src/api/`), **stateful hooks** (`src/hooks/`), and **application orchestration** (`src/App.tsx`). UI components receive state and callbacks through props.

This document explains **where state lives**, **how requests update it**, and **what happens when work is asynchronous**. It describes the implementation reviewed for this documentation pass, not planned functionality. For backend request and message formats, use the [REST API contract](../api/rest.md) and [WebSocket protocol](../api/websocket.md).

For an overview of the frontend structure, component responsibilities,
and application workflows, see [Frontend Architecture](README.md).

## Source map

| Source | Responsibility |
| --- | --- |
| `src/App.tsx` | Owns the session ID, editable board string, confirmed PLO mode, mode-change state, board/session loading, and errors; connects hooks to UI components. |
| `src/hooks/usePlayerRanges.ts` | Owns six player states, handles range and filter mutations, and applies backend-returned filter snapshots. |
| `src/hooks/useEquitySocket.ts` | Owns the equity WebSocket connection lifecycle, latest server message, UI status, and tracked job ID. |
| `src/api/client.ts` | Shared `fetch` helper for REST responses and FastAPI error messages. |
| `src/api/sessions.ts` | Session creation, PLO mode changes, and board-update REST calls. |
| `src/api/players.ts` | Player-range PUT and DELETE calls. |
| `src/api/filters.ts` | Typed filter and group REST operations that return updated player snapshots after mutations. |
| `src/api/equity.ts` | Equity WebSocket message types and connection creation. |
| `src/types.ts` | Shared mode, recursive filter-tree types, and `PlayerFilterSnapshot`. |

The UI consists of `BoardInput`, `PlayerRangeInput`, `FilterPanel`, and `EquityPanel`, coordinated by `App.tsx`.

See [Frontend Architecture](README.md#ui-components) for component responsibilities and their role in the application.

## Dependency and data-flow map

```text
App.tsx
├── local state:
│   ├── sessionId, board input, mode
│   ├── isCreatingSession, isApplyingBoard, isChangingMode
│   └── error, boardMessage
├── sessions API: createSession(), changeMode(), changeBoard()
├── usePlayerRanges(sessionId)
│   ├── six PlayerState entries (including filterSnapshot)
│   ├── range operations: add / delete
│   ├── filter operations:
│   │   ├── addFilter()
│   │   ├── addGroup()
│   │   ├── toggleFilterActive()
│   │   ├── removeFilter()
│   │   └── clearAllFilters()
│   ├── runFilterMutation() — shared mutation lifecycle
│   ├── applyBoardSnapshots()
│   ├── players API
│   └── filters API
├── useEquitySocket(sessionId)
│   ├── WebSocket connection, job ID, latest message, status
│   └── equity API: connectEquityWebSocket()
└── UI components (props + event callbacks)
    ├── BoardInput
    ├── PlayerRangeInput × 6 (left column)
    ├── FilterPanel × 0–6 (right column, applied players only)
    │   ├── AddNodeControl (local creation form state)
    │   └── FilterTreeView (local group expansion state)
    └── EquityPanel

REST clients ────> FastAPI REST routes
WebSocket client ─> FastAPI equity WebSocket route
```

The **backend owns the authoritative session, applied ranges, board, and equity job**. React maintains the state needed to edit inputs and display recent backend responses. Not all frontend state is automatically reloaded from the server.

## State ownership

| Owner | State | How it changes |
| --- | --- | --- |
| `App` | `sessionId` | Set after `createSession()` succeeds; initially `null`. |
| `App` | `board` | Updated as the user types; sent to the backend only when applied. |
| `App` | `mode` | Initialized to PLO4; updated only after `changeMode()` succeeds. The backend owns the authoritative session mode. |
| `App` | `isChangingMode` | Tracks an in-flight mode request. Disables the selector and player range inputs until the request finishes. |
| `App` | `isCreatingSession` | Tracks automatic session initialization and manual retries. Initially `true`. |
| `App` | `isApplyingBoard`, `error`, `boardMessage` | Tracks board application loading and feedback. `error` is also used for session creation failures. |
| `usePlayerRanges` | `players: PlayerState[6]` | Local edits, range-request results, and board-returned snapshots (effective hand counts and current-street filter state). |
| `AddNodeControl` | `type`, `name`, `expression`, `submitting` | Manages the creation form. `type` is `null` when closed, or `'filter'` / `'group'` when open. |
| `FilterTreeView` | `expandedGroups` | Tracks expanded direct child groups using their names. Expansion is local to each recursive component instance. |
| `useEquitySocket` | `equityStatus`, `equityMessage`, `isConnected` | WebSocket lifecycle events and incoming messages. |
| `useEquitySocket` | `socketRef`, `jobRef` | Mutable connection and job identifiers that persist across renders without triggering renders. |

Each `PlayerState` contains `range`, `numberOfHands`, `filterSnapshot`, `hasAppliedRange`, `isLoading`, and `error`.

`filterSnapshot` has type `PlayerFilterSnapshot | null` and contains the backend's latest current-street filter state. A `null` value means no snapshot is stored locally, not necessarily that the backend has no filters.

The **editable `range` string is not necessarily the range applied on the backend**: changing input text alone does not make a network request. Similarly, editing the `board` string does not change the server board until the Apply action succeeds.

`updatePlayer()` makes immutable per-slot updates using a functional React state setter. Range and filter operations update the corresponding player's loading and error state.

Filter mutations share `runFilterMutation()`, which synchronizes the effective hand count and filter snapshot using the backend response.

`App` maintains separate loading states for session creation (`isCreatingSession`), board application (`isApplyingBoard`), and mode changes (`isChangingMode`). These allow each workflow to display its own pending state and apply appropriate UI restrictions.

## REST request flows

### 1. Create a session

1. When `App` mounts, a `useEffect` automatically calls `handleCreateSession()`.
2. A `useRef` guard prevents duplicate initialization requests when React Strict Mode re-runs the effect.
3. `App` sets `isCreatingSession` to `true` and clears the previous error.
4. `createSession()` sends `POST /api/sessions` through `apiRequest()`.
5. On success, `App` stores the returned `session_id`.
6. `useEquitySocket(sessionId)` reacts to the new session ID by opening a WebSocket connection.
7. `App` resets `isCreatingSession` in `finally`.

On failure, the application displays an error message and a **Retry** button. Clicking Retry calls `handleCreateSession()` again without remounting the application.

Session creation does not block the UI with a loading indicator, but session-dependent actions remain disabled until a session exists.

Creating a session does not fetch saved player state: `usePlayerRanges()` starts with six empty local slots.

### 2. Change session mode

1. A new session starts with PLO4 as its default mode.
2. When the user selects a different mode, `App` calls `changeMode(sessionId, nextMode)`.
3. The API client sends `PUT /api/sessions/{session_id}/mode` with `{ "mode": nextMode }`.
4. While the request is pending, `isChangingMode` disables the selector and player range inputs.
5. On HTTP 204 success, `App` updates its local `mode`.
6. If the request fails, the previous mode is preserved and an error is displayed.

The selector is unavailable while any player has an applied range or is loading. The backend separately rejects changing to a different mode when player ranges exist.

Mode selection belongs to the session; subsequent player-range and equity requests use the backend's session-wide mode.

### 3. Add or replace a player range

```text
PlayerRangeInput event
    -> App callback
    -> usePlayerRanges.handleAddRange(playerId)
    -> api/players.addRange(sessionId, playerId, range)
    -> backend replaces that player's range
    -> { hands: number }
    -> update PlayerState:
         numberOfHands, hasAppliedRange,
         filterSnapshot=null, isLoading/error
    -> React rerender
```

`handleAddRange()` returns without a request if no session exists. It sets the individual player's loading state, submits the current editable range, and stores the hand count returned by the backend.

The backend interprets the range using the session-wide PLO mode; the player request does not contain a separate `mode` field. Errors are stored on the player rather than rethrown.

A successful PUT replaces that player's range and filter configurations across all existing backend street states. Consequently, `handleAddRange()` clears the local `filterSnapshot` after a successful response. The endpoint returns a hand count rather than a complete player snapshot.

Updating the local input string without applying it leaves the backend range unchanged.

### 4. Delete a player range

`handleDeleteRange()` calls the players API DELETE method. On success, it clears the local input, resets the hand count to zero, clears `filterSnapshot`, and marks the slot as not applied. On failure, it retains the previous range-related state and records an error. Empty backend slots may also be deleted successfully.

### 5. Apply or clear the board

1. `App` sets `isApplyingBoard` to `true` and clears previous feedback.
2. The board input is trimmed; an empty string becomes `null` (clear board).
3. `changeBoard(sessionId, boardValue)` sends the request to the backend.
4. On success, the backend returns `players`, an array of six current-street player snapshots or `null` values in player-slot order.
5. `App` calls `usePlayerRanges.applyBoardSnapshots(result.players)` and displays a success message.
6. The hook updates each player's `numberOfHands` from `snapshot.filtered_hands` and stores the complete `filterSnapshot` in the same React state update. Empty slots receive a count of `0` and a `null` snapshot. Range input text and applied flags are preserved.
7. In `finally`, `App` resets `isApplyingBoard` to `false`.

While the request is pending, `BoardInput` displays "Applying..." and disables its Apply Board button. Filter operations are also disabled.

If the request fails, the error is displayed and the existing player snapshots are preserved.

The backend is responsible for board validation, card blockers, street transitions, and restoring compatible cached street states.

The frontend applies the returned snapshots **only after a successful response**. It does not reconstruct or carry over filter trees between streets itself.

Session creation, board application, and mode changes use separate loading states.

### 6. Filter mutations

`usePlayerRanges()` exposes five filter-management operations:

- `addFilter()` — Creates a filter in the specified group.
- `addGroup()` — Creates a nested filter group.
- `toggleFilterActive()` — Switches a filter or group's activation state.
- `removeFilter()` — Deletes a filter or group.
- `clearAllFilters()` — Deactivates all filters and groups while preserving
  their definitions.

All mutations use `runFilterMutation()`.

The helper checks for an active session and skips the operation if the target player is loading. It then executes the REST request and updates `numberOfHands` and `filterSnapshot` using the returned snapshot.

Errors are stored in the player's state rather than propagated. The previous filter snapshot is preserved if the request fails.

`toggleFilterActive()` retrieves the target node from the locally stored filter tree and sends the opposite activation state to the backend. If the snapshot or target node is missing, the operation is skipped.

Filter mutations do not refetch the entire tree because their responses already include the updated `PlayerFilterSnapshot`.

`App.tsx` renders `FilterPanel` only for players with applied ranges, independently of the six player range inputs.

`FilterTreeView` recursively renders filters and groups. Groups are collapsed by default, and their children are mounted only when expanded. Each recursive instance manages the expansion state of its direct child groups. Collapsing a parent unmounts its descendants, resetting their expansion state.

`AddNodeControl` provides separate buttons for creating filters and groups. Clicking a button opens the corresponding form. Each instance derives its parent path from its position in the tree.

All mutations are delegated to `usePlayerRanges()` through callbacks. Creation callbacks return a success boolean, allowing the form to close and reset only after a successful operation. Errors remain managed by the hook and are displayed in `PlayerRangeInput`.

### Shared REST helper and errors

All REST clients call `apiRequest<T>()`, which uses `fetch` and:

- extracts string `detail` messages from FastAPI HTTP errors;
- joins available `msg` values from FastAPI/Pydantic validation-error arrays;
- otherwise raises an error containing HTTP status information;
- returns `undefined` for successful HTTP 204 responses;
- parses JSON for other successful responses.

The generic `T` is a **compile-time TypeScript expectation**, not runtime JSON validation. Error presentation belongs to the calling hook or `App`, not the API client.

## Equity WebSocket flow

### Connection ownership

`api/equity.ts` builds a WebSocket URL using the current page host and either `ws:` or `wss:` according to the page protocol. This assumes the equity endpoint is available on the frontend's origin (for example, through an appropriate development proxy or deployment routing).

`useEquitySocket(sessionId)` owns the socket lifecycle. On a session-ID change, its effect resets local equity state, closes the previous connection during cleanup, and opens a new connection when the ID is non-null.

The UI receives `equityStatus`, `equityMessage`, `isConnected`, `startEquity`, and `cancelEquity` from the hook. `App` passes these to `EquityPanel`.

### Start a calculation

1. The UI calls `startEquity(mode, targetMoe)`.
2. The hook sends a `start` command only if the socket is open; otherwise it returns without sending.
3. The backend uses the session's configured PLO mode and validates the calculation request.
4. On `started`, the hook records `job_id` in `jobRef` and sets status to `running`.
5. `progress` messages update the latest message and keep status `running`.
6. A terminal message updates status and clears the tracked job ID where implemented.

The mode argument remains part of the current frontend hook interface and is sent by the current client, but the backend does not use a WebSocket-supplied mode to configure equity. The authoritative mode is stored in `RangeManager`.

Sending `start` is not the same as a server acknowledgement. The hook does not mark the calculation as running until it receives `started`.

### Status transitions

| Incoming message | Current hook behavior |
| --- | --- |
| `started` | Stores `job_id`; sets `running`. |
| `resumed` | Sets `running`, even if the backend's included `status` says the job is already terminal. |
| `progress` | Stores the latest message; sets `running`. |
| `cancel_requested` | Sets `cancelling` (acknowledged request, not finished cancellation). |
| `result` with `details.stop_reason === 'cancelled'` | Sets `cancelled`; clears `jobRef`. |
| Other `result` | Sets `completed`; clears `jobRef`. |
| `cancelled` | Sets `cancelled`; clears `jobRef`. |
| `error` | Sets `error`; clears `jobRef`. |

The two cancellation paths matter: an engine that stops cooperatively may return a **`result` with `stop_reason: "cancelled"`**, whereas an explicitly cancelled future may produce a `cancelled` message. The hook's cancellation-status handling was updated to distinguish these outcomes.

The `EquityMessage` union models the expected incoming messages. The current handler checks that parsed JSON is an object with a string `type`, then asserts it is an `EquityMessage`; it does **not** validate all fields at runtime.

### Disconnect, reconnect, and resume

- An unexpected socket closure sets `isConnected` to false and schedules a retry.
- Retry delays increase as **1, 2, 4, 8, then 10 seconds maximum**; a successful connection resets the retry counter. The hook has no retry-count limit.
- Connection-open, message, and close handlers ignore events from a disposed effect or a socket no longer stored in `socketRef`.
- After reconnecting, the hook sends `resume` with `jobRef.current` **only if a job ID is still tracked**.
- Once a terminal result has cleared that ID, later connections do not automatically resume it.
- Effect cleanup clears the retry timer and closes the socket; it **does not cancel the backend job**.

The backend job belongs to the session, not the WebSocket connection. However, **the frontend job reference exists only in memory**. A page reload or remount loses `jobRef`, so this hook cannot automatically resume an earlier calculation after a full reload.

### Cancellation

`cancelEquity()` sends `cancel` only over an open socket. The method itself does not change status; the subsequent `cancel_requested` acknowledgement changes it to `cancelling`. The eventual terminal message determines whether the UI displays `cancelled`, `completed`, or `error`.

For the server-side behavior, see the [WebSocket protocol](../api/websocket.md) and [session/job lifecycle](../backend/services/session-and-jobs.md).

## Current boundaries and limitations

These are **current behavior**, not promises about planned features:

- **No automatic range-state rehydration.** `usePlayerRanges` initializes six empty slots locally and does not load the existing session's player state or reset automatically when `sessionId` changes.
- **No stale-response protection in the range hook.** Overlapping operations for one player can complete out of order; no request versioning or abort mechanism is implemented there.
- **No automatic filter-state recovery.** The hook does not refetch filter snapshots after failed or interrupted mutation responses. The backend may have completed an operation even if its response was not received.
- **No session persistence or restoration.** `App` automatically creates a new backend session on mount, but does not persist the session ID or reconnect to an existing session after a page reload. Previous session state is not automatically restored.
- **Draft inputs are distinct from applied state.** `hasAppliedRange` indicates a successful prior application, not necessarily that the current editable text equals the backend range.
- **Job resume is in-memory only.** The hook tracks a known job across reconnects within the current mount, not across page reloads.
- **WebSocket messages are only partially validated at runtime.** TypeScript assertions do not check arbitrary incoming JSON fields.
- **REST and WebSocket use the current frontend origin.** Local development and deployment must route `/api` and the equity WebSocket endpoint to FastAPI.

## Where to investigate

| Problem or question | Start with |
| --- | --- |
| A REST request returns an unexpected error | `src/api/client.ts`, then the relevant `src/api/*.ts` file and [REST contract](../api/rest.md) |
| A player hand count or filter tree is stale after changing the board | `src/App.tsx` (`handleApplyBoard`), `usePlayerRanges.applyBoardSnapshots()` |
| An applied range differs from the input field | `usePlayerRanges.ts` (`range` versus `hasAppliedRange`) |
| Equity does not start or connection status is wrong | `useEquitySocket.ts`, then `src/api/equity.ts` |
| A disconnected equity job does not resume | `useEquitySocket.ts` (`jobRef`, `onopen`, `onclose`) and [WebSocket contract](../api/websocket.md) |
| Cancellation displays the wrong status | `useEquitySocket.ts` message switch and [WebSocket contract](../api/websocket.md) |
| A filter mutation fails or returns unexpected state | `usePlayerRanges.runFilterMutation()`, then `src/api/filters.ts` and the REST contract |
| A filter control behaves incorrectly or generates the wrong parent path | `src/components/FilterPanel.tsx`, particularly `AddNodeControl()` and `FilterTreeView()` |
