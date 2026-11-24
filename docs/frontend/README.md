
# Frontend Architecture

The PLO RangeLab frontend is a React and TypeScript application built with Vite.

It provides the user interface for managing poker ranges, applying community boards, and running equity calculations against the backend.

## Structure

```text
frontend/src/
├── api/          # HTTP clients, SSE stream adapter, and message types
├── components/   # UI components
├── hooks/        # Application state and backend communication
├── test/         # Shared test setup
├── App.tsx       # Application orchestration
├── main.tsx      # React entry point
└── types.ts      # Shared frontend types
```

## Responsibilities

### Application orchestration

`App.tsx` coordinates the main application workflows.

It owns the active session ID, board input, PLO mode, and application-level loading and feedback states.

It connects the state-management hooks to the UI components and handles operations that affect multiple parts of the application.

### API clients

The `api/` directory contains typed clients for backend communication.

- `client.ts` — Shared HTTP request handling and error processing.
- `sessions.ts` — Session creation, session-wide mode changes, and board updates.
- `players.ts` — Player range requests; PUT returns a complete `PlayerFilterSnapshot`.
- `filters.ts` — Filter management requests.
- `equity.ts` — HTTP equity submission/cancellation, SSE subscription, and event types.

The REST clients handle request and response details rather than maintaining React state.

### State-management hooks

`usePlayerRanges()` manages the six player slots, including their range inputs, applied-range status, effective hand counts, current-street filter snapshots, loading indicators, and errors.

It provides operations for managing player ranges and filter trees, including creating filters and groups, toggling activation, deleting nodes, and clearing active filters.

Filter mutations share a common helper that handles loading, errors, and snapshot synchronization. Each player's range and filter snapshot are stored together in a single `PlayerState`, avoiding synchronization between separate React states.

`useEquitySSE()` manages equity job submission, the associated SSE connection, cooperative cancellation, progress messages, and calculation results. The browser handles transient SSE reconnection.

Both hooks separate application state and backend communication from UI rendering.

See [State and Data Flow](state-and-data-flow.md) for their detailed behavior.

### UI components

| Component | Responsibility |
|---|---|
| `BoardInput` | Collect board notation and request board application |
| `PlayerRangeInput` | Display and manage an individual player's range input |
| `EquityPanel` | Configure calculations and display progress and results |
| `FilterPanel` | Display filtered hand counts, coverage, and nested filter trees; provide controls for creating, activating, and deleting filters and groups |

These components receive application state and callbacks through props. They do not communicate directly with the backend.

`EquityPanel` also maintains its target margin of error input as local component state.

## Main Workflows

### Session creation

1. `App.tsx` automatically requests a backend session when the application mounts.
2. The request is sent through the sessions API.
3. On success, the returned session ID is stored in application state.
4. Session-dependent hooks use that ID to communicate with the backend.
5. On failure, the application displays an error message and a Retry button.

Session creation, board application, and mode changes maintain separate loading states.
The session ID is not displayed in the UI.

### Game mode selection

New sessions default to PLO4. The game mode selector in `App.tsx` allows switching between PLO4 and PLO5 before player ranges are applied.

Changing the selection sends `PUT /api/sessions/{session_id}/mode` through `api/sessions.ts`.

The frontend updates its local mode only after the backend confirms the change. If the request fails, the previous mode remains selected and the error is displayed.

While the mode request is pending, the selector and player range inputs are disabled. The selector is also disabled when any player has an applied range or an unfinished range operation.

The backend owns the authoritative session-wide PLO mode. Individual range submissions do not include a separate mode.

### Player range management

1. The user edits a range in `PlayerRangeInput`.
2. The change is passed to `usePlayerRanges()`.
3. Adding or deleting a range triggers a REST request.
4. The backend returns a `PlayerFilterSnapshot` containing hand counts, coverage, and the active street's filter tree (which may include automatically generated filters).
5. The hook saves the snapshot and sets `numberOfHands` from `snapshot.base_hands`.

The frontend maintains editable input state separately from whether a range has been successfully applied to the backend. Replacing a range replaces the previous filter snapshot instead of clearing it to `null`.

### Filter management

1. A filter operation is invoked through `usePlayerRanges()`.
2. The hook checks that a session exists and the player is not loading.
3. The corresponding REST client sends the mutation to the backend.
4. The backend returns a complete `PlayerFilterSnapshot`.
5. The hook updates the player's effective hand count and filter snapshot.

Supported operations include creating filters and groups, toggling activation, deleting filters or groups, and deactivating all filters.

Activation toggles determine the desired state using the locally stored filter tree. The backend remains responsible for validating and applying mutations.

`FilterPanel` provides the user interface for managing filters and groups. It renders a recursive filter tree with collapsible groups. Each node displays its coverage and activation state.

Groups are collapsed by default. Expanding a group renders its children. Expansion state is maintained locally and does not affect filter activation.

Separate `+ Filter` and `+ Group` buttons open creation forms at the root or within existing groups. Parent paths are derived automatically from the tree hierarchy.

All mutation callbacks are provided by `usePlayerRanges()` through `App.tsx`. The component does not maintain a separate copy of the filter tree.

### Board application

1. The user edits the board in `BoardInput`.
2. The input remains local until the user applies it.
3. `App.tsx` sends the board to the backend.
4. When the backend creates a new postflop street, it generates filters for occupied players; selecting a compatible cached street reuses its existing filters.
5. After a successful response, `App.tsx` passes the six returned player snapshots to `applyBoardSnapshots()`, which updates effective hand counts from `filtered_hands` and current-street filter snapshots together.

Editing the board input alone does not modify the backend session. The frontend does not generate or evaluate the automatic filter expressions.

### Equity calculations

1. The user configures the target margin of error in `EquityPanel`.
2. The component validates the input before enabling calculation.
3. The request is delegated to `useEquitySSE()`, which submits `POST /api/sessions/{session_id}/equity` with the target MOE.
4. After receiving a `job_id`, the hook opens `EventSource` on `GET /api/sessions/{session_id}/equity/{job_id}/events` and processes status/progress/result snapshots.
5. Cancellation, when requested, uses a separate HTTP POST. `EquityPanel` displays current equity, results, or errors from the hook; it does not need an SSE connection before enabling Calculate.

The equity computation itself runs in the backend.

## State Ownership

The frontend distinguishes between three kinds of state:

- **Application-level state:** Session ID, board input, confirmed game mode, mode-change loading state, and application feedback, owned by `App.tsx`.
- **Workflow state:** Player ranges and equity calculation state, managed by dedicated hooks.
- **Component-local state:** UI-specific values such as the target margin of error input.

The backend remains authoritative for the session-wide PLO mode, applied ranges, board states, filters, and equity computation.

## Current Scope

The frontend supports session creation, player range management, board application, and equity calculations.

The filter REST client and hook-level mutation operations are implemented. `usePlayerRanges()` maintains filter snapshots using player-range PUT responses, backend filter-mutation responses, and board-change responses.

`App.tsx` renders player range inputs and filter panels in two independent columns. All six range inputs are always displayed, while filter panels are rendered only for players with applied ranges.

Filter panels are ordered by their original player slot, independently of the positions of the range inputs. Each panel has a scrollable filter tree with a maximum height controlled by CSS.

The component supports nested filter and group creation, activation toggles, individual node deletion, and bulk deactivation. Filter operations are connected to `usePlayerRanges()` and covered by component tests.

Database persistence and automatic session-state restoration are not implemented.

## Further Documentation

- [Frontend State and Data Flow](state-and-data-flow.md)
- [REST API](../api/rest.md)
- [Equity SSE API](../api/sse.md)
- [Backend Architecture](../backend/README.md)
