# Backend API architecture

## Overview

The API layer exposes the in-memory PLO RangeLab services through **REST endpoints** for session, board, player-range, and filter operations, plus **HTTP submit/cancel endpoints and SSE streaming** for asynchronous equity calculations. It handles transport validation and responses; range management, filter evaluation, and equity algorithms live in the services and domain modules.

Start with this document when investigating **how an incoming request reaches application state**. For the external client contracts, see [REST API](../../api/rest.md) and [Equity SSE protocol](../../api/sse.md).

## Source map

| Source | Responsibility |
| --- | --- |
| [`main.py`](../../../src/plorangelab/api/main.py) | FastAPI application, lifespan startup/shutdown, shared executor, router registration. |
| [`dependencies.py`](../../../src/plorangelab/api/dependencies.py) | Session resolution, validated player slot indices, locked read/edit guards for REST. |
| [`schemas.py`](../../../src/plorangelab/api/schemas.py) | Pydantic request/response shapes and basic field validation. |
| [`routers/sessions.py`](../../../src/plorangelab/api/routers/sessions.py) | Session creation, session-wide PLO mode changes, and community board updates. |
| [`routers/players.py`](../../../src/plorangelab/api/routers/players.py) | Add, replace, and remove player ranges. |
| [`routers/filters.py`](../../../src/plorangelab/api/routers/filters.py) | Create, activate, clear, and delete filters/groups. |
| [`routers/equity_sse.py`](../../../src/plorangelab/api/routers/equity_sse.py) | HTTP job submission/cancellation and asynchronous SSE event streaming. |
| [`services/equity_job.py`](../../../src/plorangelab/services/equity_job.py) | Worker submission, job status, latest progress, cancellation and subscriber notifications (service layer). |

```text
FastAPI app (main.py)
├── lifespan
│   ├── load_resources() → EngineResources
│   ├── SessionManager(resources) → app.state.session_manager
│   └── ThreadPoolExecutor(max_workers=2) → app.state.executor
│
├── REST routers (sessions, players, filters)
│   └── dependencies.py → Session → RangeManager
│       └── StreetState → PlayerRange
│
└── Equity HTTP/SSE router (equity_sse.py)
    ├── POST submit/cancel → Session → EquityJob
    ├── GET events → StreamingResponse → job snapshot/subscriber
    └── start_equity_job() → application-owned executor
        └── worker → RangeManager.calculate_equity() → equity engine
```

One `SessionManager` and one application-owned executor are shared within the FastAPI application process. Each session has its **own mutable `RangeManager`**, initialized using shared engine resources, plus an optional current `EquityJob`.

## Application lifetime

`lifespan()` in `main.py` performs the following work:

1. Call `load_resources()` to prepare equity-engine resources.
2. Construct a `SessionManager` and store it on `app.state.session_manager`.
3. Construct `ThreadPoolExecutor(max_workers=2)` and store it on `app.state.executor`.

The four routers (sessions, players, filters, equity SSE) are registered on the FastAPI application at **module initialization**, separately from the lifespan context.

On application shutdown, the lifespan handler calls `session_manager.clear()` to close registered sessions and **request** cancellation of their unfinished jobs. It then calls `executor.shutdown(wait=True, cancel_futures=True)` through `asyncio.to_thread()`. Queued futures can be cancelled; already-running workers must finish cooperatively before shutdown returns.

The session registry, equity jobs, progress snapshots, and SSE subscriber callbacks are **process-local and in memory**. The current implementation does not provide persistence or cross-process synchronization for these objects.

## REST request path

A typical filter mutation follows this path:

```text
Client
└── routers/filters.py
    ├── FastAPI parses path, query, and JSON body
    ├── dependencies.get_session()
    │   └── SessionManager.get_session(session_id)
    ├── editable_session(session)
    │   ├── acquire Session.lock
    │   └── verify session is open and no unfinished equity job
    ├── session.manager.add_filter(...)
    │   └── active StreetState
    │       └── PlayerRange
    │           └── filter evaluator
    ├── RangeManager returns PlayerFilterSnapshot
    ├── PlayerFilterSnapshotResponse validates and serializes
    └── HTTP JSON response
```

`SessionDep` resolves an existing session through `SessionManager.get_session()`. A missing or expired session produces **HTTP 404**. Successful resolution refreshes session activity, but does **not** hold the lock until the endpoint returns; the endpoint explicitly enters a context manager to do that.

### Session guards

| Guard | Checks while holding `Session.lock` | REST use |
| --- | --- | --- |
| `locked_session(session)` | Session is not closed; otherwise 404. | Available for protected reads; the removed filter-read routes no longer use it. |
| `editable_session(session)` | Same closed check, plus no unfinished equity job; otherwise 409. | Mode, board, player-range, and filter mutations. |

`locked_session()` rechecks `session.closed` because deletion can occur between dependency resolution and the protected operation. `editable_session()` also prevents a running equity calculation from observing concurrent **API-mediated range mutations**. This is a convention enforced at these API entry points, not an automatic thread-safety feature of `RangeManager`.

`PlayerId` validates path indices **0–5**. Pydantic models validate fields and simple constraints; deeper checks such as card notation, filter expressions, invalid group paths, and range availability happen in the services. The REST routers generally convert service `ValueError` exceptions to HTTP 422. The mode endpoint also converts the `RuntimeError` raised when attempting to change mode with active player ranges into HTTP 409.

### Snapshot responses

Adding or replacing a player range (`PUT /api/sessions/{session_id}/players/{player_id}`) returns `PlayerFilterSnapshotResponse` for the affected player, including any automatically generated filters for the active postflop street.

Filter and group mutations also return `PlayerFilterSnapshotResponse` for the affected player.

Board changes return `BoardResponse` containing six slot-aligned player snapshots. Occupied slots contain `PlayerFilterSnapshotResponse` objects, while empty slots contain `null`. A newly created postflop street generates filters before its snapshot is returned; a reused compatible street retains its existing filter tree.

`RangeManager` constructs the service-level `PlayerFilterSnapshot` independently of HTTP. The API layer uses Pydantic response models to validate and serialize these snapshots into JSON.

See [REST API](../../api/rest.md) for the response fields and examples.

## Equity HTTP/SSE request path

The calculation API separates job submission/cancellation (HTTP POST) from updates (HTTP GET with `StreamingResponse`). The session keeps the current `EquityJob` even when a client disconnects.

```text
FastAPI application (api/main.py)
├── startup: load_resources() → EngineResources
│   ├── PLO4/PLO5 starting-hand orderings
│   └── five-card hand-ranking tables
├── HTTP routers (sessions, players, filters, equity SSE)
│   ├── session dependencies and guards
│   ├── POST equity submission
│   │   ├── validate session and player configuration
│   │   ├── start_equity_job() → thread pool
│   │   └── attach EquityJob to Session
│   ├── POST equity cancellation
│   │   └── EquityJob.cancel()
│   └── GET equity SSE stream (StreamingResponse)
│       ├── EquityJob.get_snapshot()
│       ├── subscriber notifications
│       └── progress and terminal events
└── SessionManager (in-memory registry)
    └── Session
        ├── RangeManager (mode, active street, street states)
        │   └── _streets: dict[Street, StreetState]
        │       └── StreetState (board + six player slots)
        │           └── PlayerRange | None
        └── EquityJob | None
            ├── Future[EquityResult]
            ├── cancellation event
            └── progress snapshots and subscriber

Equity calculation worker
└── RangeManager.calculate_equity()
    └── equity.compute_equity()
        ├── encoded weighted ranges and board
        ├── hand-ranking tables
        └── exact enumeration or Monte Carlo
```

The submission handler validates the session, accepts at most one unfinished job, and assigns the new `EquityJob` under `Session.lock`. The job factory does not attach the job to its session. The application-owned executor runs the CPU-intensive work, separate from the FastAPI event loop.

SSE stream generation is asynchronous. A worker's subscriber callback can run on another thread, so the router uses `loop.call_soon_threadsafe()` to wake an `asyncio.Event` before yielding new status/progress. The response can send `: heartbeat` comments during idle periods. When a job finishes, its future triggers a subscriber notification even without intermediate progress.

### Job lifetime versus stream lifetime

- **The session owns the job.** Closing an SSE connection does not itself cancel the worker.
- The stream reads the latest progress or final result and yields a terminal event once; clients call `EventSource.close()` after a terminal status to avoid automatic reconnection.
- Each job has **one subscriber callback** and a latest-value progress snapshot. A new subscription replaces the old one; this is not a broadcast mechanism or durable event log.
- `EventSource` handles temporary reconnects. A permanent connection error is handled by the current frontend by requesting HTTP cancellation and clearing its local reference.
- HTTP cancellation acknowledges the request with status **202**; it does not wait for worker completion. An immediate new submission can still receive **409** while the previous future is unfinished.

The stream and HTTP request contracts are specified in [Equity SSE protocol](../../api/sse.md).

## Concurrency and failure boundaries

| Mechanism | Protects or coordinates | Does not guarantee |
| --- | --- | --- |
| `SessionManager.lock` | In-memory session registry access. | Protection of a session's ranges. |
| `Session.lock` | Session closed/activity checks, REST guards, job association/submission/cancellation checks. | Automatic safety for arbitrary callers mutating `RangeManager`. |
| `EquityJob.lock` | Latest progress and subscriber callback. | Isolation of the worker's referenced range manager. |
| `threading.Event` | Cooperative cancellation request. | Forced termination or immediate completion. |
| `asyncio.Event` | Wake the SSE response generator when progress or terminal state changes. | A durable queue of all progress events. |
| `loop.call_soon_threadsafe()` | Forward worker-thread notifications to the event loop safely. | Delivery of every intermediate snapshot. |

Session state mutations from REST are rejected while a job has an unfinished future. The equity submit endpoint also rejects starting a second unfinished job for the same session. However, there is **no single cross-request transaction or general-purpose lock** protecting all internal service access; non-API callers must enforce their own synchronization.

The job's `Future` can finish shortly after submission. HTTP 202 and `"status": "starting"` mean **submitted**, not necessarily still running. Cancellation signals the worker; a terminal SSE outcome follows when the job actually finishes. See [Equity SSE protocol](../../api/sse.md) for the payload fields.

## Find the relevant code

| Investigating… | Start here | Further reading |
| --- | --- | --- |
| Resource initialization or shutdown | `main.py` | [Services overview](../services/README.md) |
| Session lookup / 404 / locking / 409 | `dependencies.py` | [Sessions and equity jobs](../services/session-and-jobs.md) |
| Invalid request body or response shape | `schemas.py` | [REST contract](../../api/rest.md) |
| Board and player mutations | `routers/sessions.py`, `routers/players.py` | [Range management](../services/range-management.md) |
| Nested filter path or coverage | `routers/filters.py` | [Range management](../services/range-management.md) and [filter-expression engine](../filter-engine.md) |
| Job submission, SSE stream, or reconnection | `routers/equity_sse.py` | [Equity SSE protocol](../../api/sse.md) |
| Progress notifications, cancellation, final result | `routers/equity_sse.py`, `services/equity_job.py` | [Sessions and equity jobs](../services/session-and-jobs.md) |
| Equity algorithm accuracy/performance | `RangeManager.calculate_equity()` and engine | [Equity engine](../equity.md) |

## Current scope and limitations

- Four routers are registered: session/board REST, player REST, filter REST, and equity HTTP/SSE. The current implementation does **not** expose REST endpoints to delete or enumerate sessions.
- Session expiration is lazy: `SessionManager.get_session()` checks a **30-minute** inactivity timeout, except while a job is unfinished. It does not run a periodic expiration sweep.
- No authentication, durable storage, distributed session registry, multi-worker coordination, or general broadcast subscription implementation appears in the reviewed API modules. Do not assume they are supplied by this layer.
- HTTP validation and session/job conflicts use HTTP errors; calculation failures are represented in the SSE event payload with `status: "failed"` and an `error` message.

## Related documentation

- [REST API contract](../../api/rest.md)
- [Equity SSE protocol](../../api/sse.md)
- [Services overview](../services/README.md)
- [Range management](../services/range-management.md)
- [Sessions and equity jobs](../services/session-and-jobs.md)
- [Equity engine](../equity.md)
