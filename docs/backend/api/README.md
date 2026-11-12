# Backend API architecture

## Overview

The API layer exposes the in-memory PLO RangeLab services through **REST endpoints** for session, board, player-range, and filter operations, and a **WebSocket endpoint** for asynchronous equity calculations. It handles transport validation and responses; range management, filter evaluation, and equity algorithms live in the services and domain modules.

Start with this document when investigating **how an incoming request reaches application state**. For the external client contracts, see [REST API](../../api/rest.md) and [Equity WebSocket protocol](../../api/websocket.md).

## Source map

| Source | Responsibility |
| --- | --- |
| [`main.py`](../../../src/plorangelab/api/main.py) | FastAPI application, lifespan startup/shutdown, shared executor, router registration. |
| [`dependencies.py`](../../../src/plorangelab/api/dependencies.py) | Session resolution, validated player slot indices, locked read/edit guards for REST. |
| [`schemas.py`](../../../src/plorangelab/api/schemas.py) | Pydantic request/response shapes and basic field validation. |
| [`routers/sessions.py`](../../../src/plorangelab/api/routers/sessions.py) | Session creation and shared board updates. |
| [`routers/players.py`](../../../src/plorangelab/api/routers/players.py) | Add, replace, and remove player ranges. |
| [`routers/filters.py`](../../../src/plorangelab/api/routers/filters.py) | Create, retrieve, toggle, clear, and delete filters/groups. |
| [`routers/equity_ws.py`](../../../src/plorangelab/api/routers/equity_ws.py) | WebSocket connection and `start` / `resume` / `cancel` command handling. |
| [`equity_ws_runtime.py`](../../../src/plorangelab/api/equity_ws_runtime.py) | Session-job coordination, thread-to-event-loop progress bridge, terminal message creation. |

```text
FastAPI app (main.py)
├── lifespan
│   ├── load_resources() → EngineResources
│   ├── SessionManager(resources) → app.state.session_manager
│   └── ThreadPoolExecutor(max_workers=2) → app.state.executor
│
├── REST routers (sessions, players, filters)
│   └── dependencies.py → Session → RangeManager
│       └── PlayerRangeState → FilterManager
│
└── WebSocket router (equity_ws.py)
    └── equity_ws_runtime.py
        ├── SessionManager / Session → RangeManager
        └── start_equity_job() → EquityJob → worker thread
            └── RangeManager.calculate_equity() → equity engine
```

One `SessionManager` and one application-owned executor are shared within the FastAPI application process. Each session has its **own mutable `RangeManager`**, initialized using shared engine resources, plus an optional current `EquityJob`.

## Application lifetime

`lifespan()` in `main.py` performs the following work:

1. Call `load_resources()` to prepare equity-engine resources.
2. Construct a `SessionManager` and store it on `app.state.session_manager`.
3. Construct `ThreadPoolExecutor(max_workers=2)` and store it on `app.state.executor`.

The four routers are registered on the FastAPI application at **module initialization**, separately from the lifespan context.

On application shutdown, the lifespan handler calls `session_manager.clear()` to close registered sessions and **request** cancellation of their unfinished jobs. It then calls `executor.shutdown(wait=True, cancel_futures=True)` through `asyncio.to_thread()`. Queued futures can be cancelled; already-running workers must finish cooperatively before shutdown returns.

The session registry, equity jobs, progress snapshots, and WebSocket subscriptions are **process-local and in memory**. The uploaded code does not implement persistence or cross-process synchronization for these objects.

## REST request path

A typical filter mutation follows this path:

```text
Client → routers/filters.py
          ↓ FastAPI parses path, query, JSON body
       dependencies.get_session()
          ↓ SessionManager.get_session(session_id)
       editable_session(session)
          ↓ acquire Session.lock; verify open and no unfinished job
       session.manager.add_filter(...)
          ↓ PlayerRangeState → FilterManager → filter evaluator
       Pydantic response → HTTP JSON
```

`SessionDep` resolves an existing session through `SessionManager.get_session()`. A missing or expired session produces **HTTP 404**. Successful resolution refreshes session activity, but does **not** hold the lock until the endpoint returns; the endpoint explicitly enters a context manager to do that.

### Session guards

| Guard | Checks while holding `Session.lock` | REST use |
| --- | --- | --- |
| `locked_session(session)` | Session is not closed; otherwise 404. | Filter reads. |
| `editable_session(session)` | Same closed check, plus no unfinished equity job; otherwise 409. | Board, range, and filter writes. |

`locked_session()` rechecks `session.closed` because deletion can occur between dependency resolution and the protected operation. `editable_session()` also prevents a running equity calculation from observing concurrent **API-mediated range mutations**. This is a convention enforced at these API entry points, not an automatic thread-safety feature of `RangeManager`.

`PlayerId` validates path indices **0–5**. Pydantic models validate fields and simple constraints; deeper checks such as card notation, filter expressions, invalid group paths, and range availability happen in the services. The REST routers generally convert service `ValueError` to **HTTP 422**. Read endpoints for missing filters instead convert those errors to **HTTP 404**.

## WebSocket request path

The endpoint is `WS /api/sessions/{session_id}/equity/ws`. Unlike REST, it resolves the session directly through the application state and performs blocking session operations through `asyncio.to_thread()`.

```text
Client WebSocket
    → routers/equity_ws.py: command dispatcher
        → equity_ws_runtime.submit_equity_calculation()
            → Session.lock: validate and attach EquityJob
            → start_equity_job(session.manager, app.state.executor, ...)
                → worker: RangeManager.calculate_equity()
                    → EquityJob.update_progress()
                        → callback → loop.call_soon_threadsafe(event.set)
        → publish_equity_updates() task
            → read latest job snapshot / final Future
            → serialize and send JSON using connection send lock
```

The router interprets `start`, `resume`, and `cancel`. The runtime handles job ownership, status/progress retrieval, error-to-message conversion, and cooperative cancellation. A successful `start` command creates a job and assigns it to `session.equity_job` **under the session lock**; `start_equity_job()` by itself does not assign jobs to sessions.

`asyncio.to_thread()` offloads synchronous operations using the event loop's thread-execution facilities. This is distinct from the **explicit two-worker executor** in `app.state.executor`, which runs the equity calculations.

### Job lifetime versus connection lifetime

- The **session** retains the current job. A WebSocket does not own its worker.
- Closing a WebSocket cancels its **publisher task**, not the equity calculation. A later connection can `resume` the stored job, including an already-completed one.
- Each job retains **one subscriber callback** and its latest progress snapshot. A new publisher replaces the previous subscriber; intermediate progress is not queued or guaranteed to be delivered.
- The worker updates job state under a regular thread lock. Its callback uses `loop.call_soon_threadsafe()` to signal an `asyncio.Event`; only the asynchronous publisher sends WebSocket messages.
- An `asyncio.Lock` serializes messages sent by a connection's command handler and its publisher task.

The router rechecks session validity and refreshes session activity on **each received command**. It closes with WebSocket code **1008** when the session is unavailable. Simply holding an open, idle WebSocket connection is not a session heartbeat in this implementation.

## Concurrency and failure boundaries

| Mechanism | Protects or coordinates | Does not guarantee |
| --- | --- | --- |
| `SessionManager.lock` | In-memory session registry access. | Protection of a session's ranges. |
| `Session.lock` | Session closed/activity checks, REST guards, job association/submission/cancellation checks. | Automatic safety for arbitrary callers mutating `RangeManager`. |
| `EquityJob.lock` | Latest progress and subscriber callback. | Isolation of the worker's referenced range manager. |
| `threading.Event` | Cooperative cancellation request. | Forced termination or immediate completion. |
| `asyncio.Event` | Wake up a WebSocket publisher on its event loop. | A durable queue of all progress events. |
| `asyncio.Lock` (`send_lock`) | Serializes sends on one WebSocket. | Guaranteed delivery to disconnected clients. |

Session state mutations from REST are rejected while a job has an unfinished future. The WebSocket runtime also rejects starting a second unfinished job for the same session. However, there is **no single cross-request transaction or general-purpose lock** protecting all internal service access; non-API callers must enforce their own synchronization.

The job's `Future` can finish after submission but before the client receives `started`; the acknowledgment means **submitted**, not necessarily still running. Cancellation signals the worker; a terminal outcome follows when the job actually finishes. See [WebSocket protocol](../../api/websocket.md) for the precise message types.

## Find the relevant code

| Investigating… | Start here | Further reading |
| --- | --- | --- |
| Resource initialization or shutdown | `main.py` | [Services overview](../services/README.md) |
| Session lookup / 404 / locking / 409 | `dependencies.py` | [Sessions and equity jobs](../services/session-and-jobs.md) |
| Invalid request body or response shape | `schemas.py` | [REST contract](../../api/rest.md) |
| Board and player mutations | `routers/sessions.py`, `routers/players.py` | [Range management](../services/range-management.md) |
| Nested filter path or coverage | `routers/filters.py` | [Range management](../services/range-management.md) and [filter-expression engine](../filter-engine.md) |
| WebSocket commands or reconnect | `routers/equity_ws.py` | [WebSocket protocol](../../api/websocket.md) |
| Progress notifications, cancellation, final result | `equity_ws_runtime.py` | [Sessions and equity jobs](../services/session-and-jobs.md) |
| Equity algorithm accuracy/performance | `RangeManager.calculate_equity()` and engine | [Equity engine](../equity.md) |

## Current scope and limitations

- Four routers are registered: session/board REST, player REST, filter REST, and equity WebSocket. The uploaded code does **not** expose REST endpoints to delete or enumerate sessions.
- Session expiration is lazy: `SessionManager.get_session()` checks a **30-minute** inactivity timeout, except while a job is unfinished. It does not run a periodic expiration sweep.
- No authentication, durable storage, distributed session registry, multi-worker coordination, or general broadcast subscription implementation appears in the reviewed API modules. Do not assume they are supplied by this layer.
- WebSocket command-level errors are JSON `error` messages; invalid JSON syntax is not explicitly caught in the router's receive loop.

## Related documentation

- [REST API contract](../../api/rest.md)
- [Equity WebSocket protocol](../../api/websocket.md)
- [Services overview](../services/README.md)
- [Range management](../services/range-management.md)
- [Sessions and equity jobs](../services/session-and-jobs.md)
- [Equity engine](../equity.md)
