# Frontend state and data flow

## Overview

PLO RangeLab's React frontend coordinates session setup, player-range editing, community board changes, and live equity calculations. The current implementation separates **transport functions** (`src/api/`), **stateful hooks** (`src/hooks/`), and **application orchestration** (`src/App.tsx`). UI components receive state and callbacks through props.

This document explains **where state lives**, **how requests update it**, and **what happens when work is asynchronous**. It describes the implementation reviewed for this documentation pass, not planned functionality. For backend request and message formats, use the [REST API contract](../api/rest.md) and [Equity SSE protocol](../api/sse.md).

For an overview of the frontend structure, component responsibilities,
and application workflows, see [Frontend Architecture](README.md).

## Source map

| Source | Responsibility |
| --- | --- |
| `src/App.tsx` | Owns the session ID, editable board string, confirmed PLO mode, mode-change state, board/session loading, and errors; connects hooks to UI components. |
| `src/hooks/usePlayerRanges.ts` | Owns six player states, handles range and filter mutations, and applies backend-returned filter snapshots. |
| `src/hooks/useEquitySSE.ts` | Owns equity job submission/cancellation, the SSE connection lifecycle, latest server message, UI status, and tracked job ID. |
| `src/api/client.ts` | Shared `fetch` helper for REST responses and FastAPI error messages. |
| `src/api/sessions.ts` | Session creation, PLO mode changes, and board-update REST calls. |
| `src/api/players.ts` | Player-range PUT (returns `PlayerFilterSnapshot`) and DELETE calls. |
| `src/api/filters.ts` | Typed filter and group REST operations that return updated player snapshots after mutations. |
| `src/api/equity.ts` | HTTP equity submission/cancellation, `EventSource` creation and parsing, SSE event types and failure callbacks. |
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
├── useEquitySSE(sessionId)
│   ├── EventSource, job ID ref, latest message, status/error
│   └── equity API: submitEquityCalculation(), startEquityStream(),
│                   cancelEquityCalculation()
└── UI components (props + event callbacks)
    ├── BoardInput
    ├── PlayerRangeInput × 6 (left column)
    ├── FilterPanel × 0–6 (right column, applied players only)
    │   ├── AddNodeControl (local creation form state)
    │   └── FilterTreeView (local group expansion state)
    └── EquityPanel

HTTP clients ────> FastAPI REST + equity POST endpoints
EventSource GET ─> FastAPI equity SSE stream
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
| `useEquitySSE` | `equityStatus`, `equityMessage`, `equityError` | Job submissions, incoming SSE snapshots, terminal outcomes, and errors. |
| `useEquitySSE` | `sourceRef`, `jobIdRef`, `generationRef` and other guard refs | Stream/job identity and stale-async-response protection without rerendering. |

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
6. `useEquitySSE(sessionId)` resets equity state for the new session ID; it opens a stream **only when a calculation is submitted**.
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
    -> PlayerFilterSnapshot (base_hands, filtered_hands,
                             filtered_coverage, filters)
    -> update PlayerState:
         numberOfHands=snapshot.base_hands, hasAppliedRange,
         filterSnapshot=snapshot, isLoading/error
    -> React rerender
```

`handleAddRange()` returns without a request if no session exists. It sets the individual player's loading state, submits the current editable range, then sets `numberOfHands` from the returned snapshot's `base_hands` and saves the full `filterSnapshot`.

The backend interprets the range using the session-wide PLO mode; the player request does not contain a separate `mode` field. Errors are stored on the player rather than rethrown.

A successful PUT replaces that player's range and filter configurations across all existing backend street states. The response is a complete player snapshot, not `{ "hands": number }`. If a board is active, its new filter tree may contain automatically generated nodes. The hook replaces the previous local `filterSnapshot` with this returned snapshot, including when replacing an existing range.

`numberOfHands` on range PUT comes from `base_hands`; after a board change or filter mutation, it is updated from `filtered_hands`. This distinction matters if the returned snapshot has active filtering.

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

The backend is responsible for board validation, card blockers, street transitions, generating filters on new postflop states, and restoring compatible cached street states without regenerating their filters.

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

## Equity HTTP and SSE flow

### Connection ownership

`api/equity.ts` submits jobs and requests cancellation using `apiRequest()`, and opens a native `EventSource` using a same-origin `/api/sessions/{session_id}/equity/{job_id}/events` URL. Development and deployment must route `/api` to FastAPI. The backend uses the session's configured PLO mode; neither the hook nor the HTTP submission needs a separate mode argument.

`useEquitySSE(sessionId)` owns job and stream lifecycle. It tracks `sourceRef`, `jobIdRef`, `submittingRef`, `cancellingRef`, `cancellationRequestedRef`, and `generationRef` for request coordination. React state includes `equityStatus`, `equityMessage`, and `equityError`. `App` passes these to `EquityPanel`, which receives `hasSession`, not an `isConnected` requirement.

A browser SSE connection is created **after** the job is accepted, not on session creation. The Calculate action is enabled when a session exists, the target MOE is valid, and the UI is not in `starting`, `running`, or `cancelling` state.

### Start and progress

1. `EquityPanel` calls `startEquity(targetMoe)`.
2. The hook rejects duplicate local submissions and marks state `starting`.
3. `startEquityStream()` POSTs `{"target_moe": targetMoe}` to `/api/sessions/{session_id}/equity` and receives HTTP 202 plus `job_id`.
4. It opens an `EventSource` for that job; the backend emits default SSE events with JSON in `event.data`.
5. The API layer parses each event into an `EquityStreamEvent`; the hook stores the latest snapshot and maps `pending` / `running` to the frontend `running` state.
6. On `completed`, `cancelled`, or `failed`, the hook records the terminal message, clears the tracked job and stream refs, and reports errors when appropriate. The API adapter closes `EventSource` on terminal status.

The API event object carries `job_id`, `status`, `error`, and `details`. `details` may contain `equities`, `margins_of_error`, `method`, `simulations`, `attempts`, `stop_reason`, `sampler`, and `effective_samples`; it may be null before progress is available. Player IDs remain **string keys in JSON**. `EquityPanel` shows both progress and terminal equities from `details`, ordering them numerically by player slot and displaying MOEs when present.

### Status transitions

| Input or event | Current hook behavior |
| --- | --- |
| Start action | `starting` while awaiting HTTP submission. |
| SSE `pending` or `running` | `running`, except while cancellation is already requested. |
| SSE `completed` | `completed`; retain result and release job reference. |
| SSE `cancelled` | `cancelled`; retain any returned partial result and release job reference. |
| SSE `failed` | `failed` with backend error and released job reference. |
| Cancellation POST accepted (202) | `cancelling`; waits for terminal SSE event. |
| Submission POST fails | `failed` with HTTP error; no SSE stream established. |
| Unrecoverable SSE/parse error | `cancelling` while requesting backend cancellation, then `failed` and clear local job reference. |

The backend may complete normally while cancellation is being requested; the terminal event, not the HTTP 202 acknowledgment, determines the final calculation outcome.

### Disconnect, reconnect, and cleanup

- Native `EventSource` retries **temporary** connection interruptions automatically. The frontend does not implement a manual exponential-backoff WebSocket loop.
- When `EventSource` becomes permanently `CLOSED`, or the client cannot parse SSE JSON, the API layer calls `onStreamError`.
- `useEquitySSE` routes both cases through its shared `handleStreamFailure()` helper: close the stream, request cancellation through HTTP, clear the stored job reference, and display an error. The UI does **not** show a manual Reconnect button.
- The backend can still be completing a cooperatively cancelled job after HTTP 202. An immediate fresh submission may receive HTTP 409 until that future is finished.
- On session change or hook unmount, the hook closes its stream and increments `generationRef` to ignore late callbacks or requests. Closing a stream **does not cancel** a backend calculation on its own.
- A page reload creates a new local session under the current `App` design, and the hook does not persist or restore old job IDs.

The backend retains only the latest progress snapshot and a single subscriber callback, not an event history. Reconnecting streams may receive a newer snapshot rather than all missed events.

### Cancellation

`cancelEquity()` sends `POST /api/sessions/{session_id}/equity/{job_id}/cancel`. It can be requested while a calculation is `running` without requiring the SSE stream to be connected; the stream stays open until a terminal status. Duplicate cancellation requests are guarded locally. On a failed cancellation HTTP request, the hook retains the running job and displays an error.

For the wire-level schema, see [Equity SSE protocol](../api/sse.md). For job state, subscriptions, and cancellation semantics, see [Sessions and equity jobs](../backend/services/session-and-jobs.md).

## Current boundaries and limitations

These are **current behavior**, not promises about planned features:

- **No automatic range-state rehydration.** `usePlayerRanges` initializes six empty slots locally and does not load the existing session's player state or reset automatically when `sessionId` changes.
- **No stale-response protection in the range hook.** Overlapping operations for one player can complete out of order; no request versioning or abort mechanism is implemented there.
- **No automatic filter-state recovery.** The hook does not refetch filter snapshots after failed or interrupted mutation responses. The backend may have completed an operation even if its response was not received.
- **No session persistence or restoration.** `App` automatically creates a new backend session on mount, but does not persist the session ID or reconnect to an existing session after a page reload. Previous session state is not automatically restored.
- **Draft inputs are distinct from applied state.** `hasAppliedRange` indicates a successful prior application, not necessarily that the current editable text equals the backend range.
- **No durable job restoration.** The hook tracks the current job only in memory and does not restore it across a reload; temporary `EventSource` retries are browser-managed.
- **SSE payloads are only partially validated at runtime.** JSON is parsed, but TypeScript assertions do not verify every incoming field.
- **HTTP and SSE use the current frontend origin.** Local development and deployment must route `/api` to FastAPI.

## Where to investigate

| Problem or question | Start with |
| --- | --- |
| A REST request returns an unexpected error | `src/api/client.ts`, then the relevant `src/api/*.ts` file and [REST contract](../api/rest.md) |
| A player hand count or filter tree is stale after adding a range or changing the board | `usePlayerRanges.handleAddRange()` / `applyBoardSnapshots()`, then `src/api/players.ts` / `sessions.ts` |
| An applied range differs from the input field | `usePlayerRanges.ts` (`range` versus `hasAppliedRange`) |
| Equity does not start or its status is wrong | `useEquitySSE.ts`, then `src/api/equity.ts` and [SSE contract](../api/sse.md) |
| SSE disconnects or a malformed event is received | `src/api/equity.ts` (`onStreamError`), then `useEquitySSE.ts` (`handleStreamFailure`) |
| Cancellation displays the wrong status | `useEquitySSE.ts` terminal-event handler and [SSE contract](../api/sse.md) |
| A filter mutation fails or returns unexpected state | `usePlayerRanges.runFilterMutation()`, then `src/api/filters.ts` and the REST contract |
| A filter control behaves incorrectly or generates the wrong parent path | `src/components/FilterPanel.tsx`, particularly `AddNodeControl()` and `FilterTreeView()` |
