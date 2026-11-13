# Frontend state and data flow

## Overview

PLO RangeLab's React frontend coordinates session setup, player-range editing, community board changes, and live equity calculations. The current implementation separates **transport functions** (`src/api/`), **stateful hooks** (`src/hooks/`), and **application orchestration** (`src/App.tsx`). UI components receive state and callbacks through props.

This document explains **where state lives**, **how requests update it**, and **what happens when work is asynchronous**. It describes the implementation reviewed for this documentation pass, not planned functionality. For backend request and message formats, use the [REST API contract](../api/rest.md) and [WebSocket protocol](../api/websocket.md).

For an overview of the frontend structure, component responsibilities,
and application workflows, see [Frontend Architecture](README.md).

## Source map

| Source | Responsibility |
| --- | --- |
| `src/App.tsx` | Owns the session ID, editable board string, selected PLO mode, board/session loading and errors; connects hooks to UI components. |
| `src/hooks/usePlayerRanges.ts` | Owns six local player states and handles add/replace/delete range requests. |
| `src/hooks/useEquitySocket.ts` | Owns the equity WebSocket connection lifecycle, latest server message, UI status, and tracked job ID. |
| `src/api/client.ts` | Shared `fetch` helper for REST responses and FastAPI error messages. |
| `src/api/sessions.ts` | Session creation and board-update REST calls. |
| `src/api/players.ts` | Player-range PUT and DELETE calls. |
| `src/api/filters.ts` | Filter-tree types and REST calls; **not yet wired into React state or UI**. |
| `src/api/equity.ts` | Equity WebSocket message types and connection creation. |
| `src/types.ts` | Shared `Mode = 4 | 5` type. |

The current UI consists of `BoardInput`, `PlayerRangeInput`, and `EquityPanel`, coordinated by `App.tsx`. Their internal rendering and props will be covered in the frontend architecture overview.

## Dependency and data-flow map

```text
App.tsx
├── local state: sessionId, board input, mode, loading/errors, board message
├── sessions API: createSession(), changeBoard()
├── usePlayerRanges(sessionId, mode)
│   ├── six PlayerState entries
│   └── players API: addRange(), deleteRange()
├── useEquitySocket(sessionId)
│   ├── WebSocket connection, job ID, latest message, status
│   └── equity API: connectEquityWebSocket()
└── UI components (props + event callbacks)
    ├── BoardInput
    ├── PlayerRangeInput × 6
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
| `App` | `mode` | User chooses PLO4 or PLO5; the selector is disabled while any player has an applied range or is loading. |
| `App` | `isLoading`, `error`, `boardMessage` | Session creation and board-request feedback. |
| `usePlayerRanges` | `players: PlayerState[6]` | Local edits, range-request results, and board hand-count updates. |
| `useEquitySocket` | `equityStatus`, `equityMessage`, `isConnected` | WebSocket lifecycle events and incoming messages. |
| `useEquitySocket` | `socketRef`, `jobRef` | Mutable connection and job identifiers that persist across renders without triggering renders. |

Each `PlayerState` contains `range`, `numberOfHands`, `hasAppliedRange`, `isLoading`, and `error`. The **editable `range` string is not necessarily the range applied on the backend**: changing input text alone does not make a network request. Similarly, editing the `board` string does not change the server board until the Apply action succeeds.

`updatePlayer()` makes immutable per-slot updates using a functional React state setter. Individual range operations maintain their own loading and error state; `App` maintains a separate loading flag for session and board operations.

## REST request flows

### 1. Create a session

1. The user requests session creation in `App`.
2. `App` clears its error and sets `isLoading`.
3. `createSession()` sends `POST /api/sessions` through `apiRequest()`.
4. On success, `App` stores the returned `session_id`.
5. `useEquitySocket(sessionId)` reacts to the new session ID by opening a WebSocket.
6. `App` clears its loading flag in `finally`; on failure it displays the error.

Creating a session does not fetch saved player state: `usePlayerRanges` starts with six empty local slots.

### 2. Add or replace a player range

```text
PlayerRangeInput event
    -> App callback
    -> usePlayerRanges.handleAddRange(playerId)
    -> api/players.addRange(sessionId, playerId, range, mode)
    -> backend replaces that player's range
    -> { hands: number }
    -> update PlayerState: numberOfHands, hasAppliedRange, isLoading/error
    -> React rerender
```

`handleAddRange()` returns without a request if no session exists. It sets the individual player's loading state, uses the current editable range and mode, then stores the hand count returned by the backend. Errors are stored on that player rather than rethrown.

A successful PUT **replaces** backend player state, including any previous street/filter state. Updating the local input string without applying it leaves the backend range unchanged.

### 3. Delete a player range

`handleDeleteRange()` calls the players API DELETE method. On success it clears the local input, resets the hand count to zero, and marks the slot as not applied. On failure it retains the previous range-related state and records an error. Empty backend slots may also be deleted successfully.

### 4. Apply or clear the board

1. `App` takes the trimmed board input; an empty string becomes `null` (clear board).
2. `changeBoard(sessionId, boardValue)` sends the request.
3. On success, the backend returns `players`, an array of **six hand counts in slot order**.
4. `App` shows a success message and calls `usePlayerRanges.updateHandCounts(result.players)`.
5. The hook updates the counts while preserving range text, applied flags, and other player fields.

The backend is responsible for board validation, blockers, and street-state changes. The frontend updates hand counts **only after a successful response**.

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
2. The hook sends a `start` JSON command **only if the socket is open**; otherwise it returns without sending.
3. The backend validates the request and submits an equity job.
4. On `started`, the hook records `job_id` in `jobRef` and sets status to `running`.
5. `progress` messages update the latest message and keep status `running`.
6. A terminal message updates status and clears the tracked job ID where implemented.

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

- **Frontend filtering is not yet integrated.** `api/filters.ts` contains typed request functions and recursive tree definitions, but the uploaded `App` and hooks do not yet manage filter state or expose filter UI workflows.
- **No automatic range-state rehydration.** `usePlayerRanges` initializes six empty slots locally and does not load the existing session's player state or reset automatically when `sessionId` changes.
- **No stale-response protection in the range hook.** Overlapping operations for one player can complete out of order; no request versioning or abort mechanism is implemented there.
- **Draft inputs are distinct from applied state.** `hasAppliedRange` indicates a successful prior application, not necessarily that the current editable text equals the backend range.
- **Job resume is in-memory only.** The hook tracks a known job across reconnects within the current mount, not across page reloads.
- **WebSocket messages are only partially validated at runtime.** TypeScript assertions do not check arbitrary incoming JSON fields.
- **REST and WebSocket use the current frontend origin.** Local development and deployment must route `/api` and the equity WebSocket endpoint to FastAPI.

## Where to investigate

| Problem or question | Start with |
| --- | --- |
| A REST request returns an unexpected error | `src/api/client.ts`, then the relevant `src/api/*.ts` file and [REST contract](../api/rest.md) |
| A player hand count is stale after changing the board | `src/App.tsx` (`handleApplyBoard`), `usePlayerRanges.updateHandCounts()` |
| An applied range differs from the input field | `usePlayerRanges.ts` (`range` versus `hasAppliedRange`) |
| Equity does not start or connection status is wrong | `useEquitySocket.ts`, then `src/api/equity.ts` |
| A disconnected equity job does not resume | `useEquitySocket.ts` (`jobRef`, `onopen`, `onclose`) and [WebSocket contract](../api/websocket.md) |
| Cancellation displays the wrong status | `useEquitySocket.ts` message switch and [WebSocket contract](../api/websocket.md) |
| Filter API calls exist but no filter UI appears | `src/api/filters.ts`; React integration is not yet implemented |

