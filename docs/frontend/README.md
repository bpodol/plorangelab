
# Frontend Architecture

The PLO RangeLab frontend is a React and TypeScript application built with Vite.

It provides the user interface for managing poker ranges, applying community boards, and running equity calculations against the backend.

## Structure

```text
frontend/src/
├── api/          # REST clients and WebSocket message types
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
- `sessions.ts` — Session creation and management requests.
- `players.ts` — Player range requests.
- `filters.ts` — Filter management requests.
- `equity.ts` — Equity calculation message types.

The REST clients handle request and response details rather than maintaining React state.

### State-management hooks


`usePlayerRanges()` manages the six player slots, including their range inputs, applied-range status, effective hand counts, current-street filter snapshots, loading indicators, and errors.

Each player's range and filter snapshot are stored together in a single `PlayerState`, avoiding synchronization between separate React states.


`useEquitySocket()` manages the session's WebSocket connection, reconnection, equity job lifecycle, progress messages, and calculation results.

Both hooks separate application state and backend communication from UI rendering.

See [State and Data Flow](state-and-data-flow.md) for their detailed behavior.

### UI components

| Component | Responsibility |
|---|---|
| `BoardInput` | Collect board notation and request board application |
| `PlayerRangeInput` | Display and manage an individual player's range input |
| `EquityPanel` | Configure calculations and display progress and results |

These components receive application state and callbacks through props. They do not communicate directly with the backend.

`EquityPanel` also maintains its target margin of error input as local component state.

## Main Workflows

### Session creation

1. The user requests a new session.
2. `App.tsx` calls the sessions API.
3. The returned session ID is stored in application state.
4. Session-dependent hooks use that ID to communicate with the backend.

### Player range management

1. The user edits a range in `PlayerRangeInput`.
2. The change is passed to `usePlayerRanges()`.
3. Adding or deleting a range triggers a REST request.
4. The hook updates the corresponding player state using the backend response.

The frontend maintains editable input state separately from whether a range has been successfully applied to the backend.

### Board application

1. The user edits the board in `BoardInput`.
2. The input remains local until the user applies it.
3. `App.tsx` sends the board to the backend.
4. After a successful response, `App.tsx` passes the six returned player snapshots to `applyBoardSnapshots()`, which updates effective hand counts and current-street filter snapshots together.

Editing the board input alone does not modify the backend session.

### Equity calculations

1. The user configures the target margin of error in `EquityPanel`.
2. The component validates the input before enabling calculation.
3. The calculation request is delegated to `useEquitySocket()`.
4. The hook processes WebSocket messages and maintains calculation state.
5. `EquityPanel` renders progress, results, or errors from that state.

The equity computation itself runs in the backend.

## State Ownership

The frontend distinguishes between three kinds of state:

- **Application-level state:** Session ID, board input, game mode, and application feedback, owned by `App.tsx`.
- **Workflow state:** Player ranges and equity calculation state, managed by dedicated hooks.
- **Component-local state:** UI-specific values such as the target margin of error input.

The backend remains authoritative for applied ranges, board state, and equity computation.

## Current Scope

The frontend currently supports session creation, player range management, board application, and equity calculations.

The filter REST client is implemented, and `usePlayerRanges()` stores current-street filter snapshots returned by board changes.

Filter mutation actions and filter-management components are not yet integrated into the frontend UI.

## Further Documentation

- [Frontend State and Data Flow](state-and-data-flow.md)
- [REST API](../api/rest.md)
- [WebSocket API](../api/websocket.md)
- [Backend Architecture](../backend/README.md)
