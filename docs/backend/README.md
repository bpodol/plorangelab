# Backend architecture

## Overview

PLO RangeLab's Python backend accepts PLO4/PLO5 player ranges, community boards, and card-based filters, then calculates multiway equity using exact enumeration or Monte Carlo estimation. It separates **transport and request validation**, **session and range state**, and **domain algorithms** so each layer has a clear responsibility.

This page is the **entry point for investigating the backend codebase**. It maps the components and common execution paths to their detailed documentation; it does not repeat their implementations or the client-facing API contracts.

## Documentation map

| If you want to understand… | Read |
| --- | --- |
| How FastAPI initializes resources and handles REST/WebSocket requests | [API architecture](api/README.md) |
| Which service owns sessions, player ranges, filters, and calculation jobs | [Services overview](services/README.md) |
| Player slots, board transitions, per-street ranges, filter trees, and weighted coverage | [Range management](services/range-management.md) |
| Session expiry, equity job status, worker execution, progress, and cancellation | [Sessions and equity jobs](services/session-and-jobs.md) |
| Card IDs, board masks, `EncodedRange`, weighted ranges, and percentage-based ranges | [Card and range encoding](encoding.md) |
| Filter expression grammar, compilation, and Boolean hand selections | [Filter expression engine](filter-engine.md) |
| Five-card strength scores, ranking tables, and combinatorial lookup | [Five-card hand ranking](hand-ranking.md) |
| PLO showdown evaluation, exact equity, Monte Carlo samplers, and diagnostics | [Equity engine](equity.md) |

**For client integration**, use the separate [REST API contract](../api/rest.md) and [Equity WebSocket protocol](../api/websocket.md). Those documents specify routes, JSON messages, validation, and client-visible behavior.

## System map

```text
FastAPI application (api/main.py)
├── startup: load_resources() -> EngineResources
│   ├── PLO4/PLO5 starting-hand orderings
│   └── five-card hand-ranking tables
├── REST routers (sessions, players, filters)
│   └── session dependencies and guards
├── equity WebSocket router
│   └── equity_ws_runtime.py
│       └── start_equity_job() -> application-owned thread pool
└── SessionManager (in-memory registry)
    └── Session
        ├── RangeManager (up to six player slots and shared board)
        │   └── PlayerRangeState
        │       └── StreetState -> FilterManager
        └── EquityJob | None (future, cancellation, progress)

EquityJob worker
    -> RangeManager.calculate_equity()
    -> equity.compute_equity()
        -> encoded weighted ranges + board + ranking tables
        -> exact enumeration or Monte Carlo
```

`EngineResources` is prepared at application startup and shared by session range managers. Mutable player, board, and filter state belongs to individual sessions. The WebSocket runtime attaches jobs to sessions; the job factory itself does not manage session ownership.

## Follow an equity calculation

1. **Start a session.** The REST API creates a `Session`, which owns an initially empty `RangeManager`.
2. **Prepare the inputs.** REST calls populate player slots, set or clear the board, and create or activate filters. The services maintain each player's current weighted range for the active street.
3. **Submit work.** A WebSocket `start` command validates the session and player configuration, attaches an `EquityJob`, and submits `RangeManager.calculate_equity()` to the application's thread pool.
4. **Compute equity.** `RangeManager` collects occupied players' current ranges. The equity engine validates and evaluates them with exact enumeration or Monte Carlo, using precomputed hand-ranking resources.
5. **Deliver updates.** The worker records progress on `EquityJob`. The WebSocket publisher reads the latest snapshot and sends progress or the terminal result through the async connection.
6. **Resume or cancel.** The job belongs to the session, not the socket. Reconnecting clients can resume it; cancellation requests signal the worker cooperatively.

For the detailed request paths and lock boundaries, see [API architecture](api/README.md). For ownership and job lifecycles, see the [services overview](services/README.md).

## Key boundaries

- **Encoding versus ranking:** `encoder.py` converts card and range notation into compact representations. `rating.py` loads externally ordered *starting hands* for percentage ranges; `rank_combination.py` constructs *five-card showdown* ranking tables. These are different resources.
- **Filter engine versus filter state:** The `filters` package evaluates expressions and returns Boolean hand selections. `FilterManager` owns named filters, nested groups, activation, weighted coverage, and the derived current range.
- **Services versus computation:** `RangeManager` owns mutable player and board state, but delegates equity algorithms to the `equity` package. `EquityJob` schedules and observes calculations without implementing the algorithms.
- **API versus services:** REST and WebSocket handlers validate requests, guard access to session state, and serialize results. They rely on services for state changes and on domain modules for evaluation.
- **Concurrency versus failure atomicity:** Clone-based range updates avoid publishing partially modified state after a failed operation. Session locks and REST mutation guards coordinate API-mediated access; they are not a general guarantee that every backend object is safe for arbitrary concurrent mutation.

## Where to investigate a problem

| Symptom or question | Start with |
| --- | --- |
| A request returns HTTP 404, 409, or 422 | [API architecture](api/README.md), then [REST API contract](../api/rest.md) |
| A WebSocket starts, disconnects, or resumes unexpectedly | [WebSocket protocol](../api/websocket.md), then [API architecture](api/README.md) |
| A session expires or a calculation does not stop immediately | [Sessions and equity jobs](services/session-and-jobs.md) |
| A board change resets filters or changes the range size | [Range management](services/range-management.md) |
| A filter expression matches unexpected hands | [Filter expression engine](filter-engine.md) |
| Filter activation or coverage is surprising | [Range management](services/range-management.md) |
| Percentage range selection looks wrong | [Card and range encoding](encoding.md) |
| Showdown strength or hand ordering looks wrong | [Five-card hand ranking](hand-ranking.md), then [Equity engine](equity.md) |
| Equity precision, sample counts, or execution time needs investigation | [Equity engine](equity.md) |

## Source directories

| Location | Primary role |
| --- | --- |
| [`src/plorangelab/api/`](../../src/plorangelab/api/) | FastAPI app, dependencies, schemas, routers, and WebSocket runtime |
| [`src/plorangelab/services/`](../../src/plorangelab/services/) | Session lifecycle, range state, filtering state, and equity jobs |
| [`src/plorangelab/equity/`](../../src/plorangelab/equity/) | Equity calculation algorithms and results |
| [`src/plorangelab/filters/`](../../src/plorangelab/filters/) | Stateless filter parsing, compilation, and evaluation |
| [`src/plorangelab/encoder.py`](../../src/plorangelab/encoder.py) and [`rating.py`](../../src/plorangelab/rating.py) | Card/range encoding and preflop starting-hand orderings |
| [`src/plorangelab/rank_combination.py`](../../src/plorangelab/rank_combination.py) and [`bootstrap.py`](../../src/plorangelab/bootstrap.py) | Five-card rank tables and shared engine resource initialization |

The current API uses process-local sessions and an application-owned thread pool. See the subsystem documents for specific limitations and implementation guarantees; this overview is intentionally a map, not a specification for persistence, authentication, or deployment.
