# Backend architecture

## Overview

PLO RangeLab's Python backend accepts PLO4/PLO5 player ranges, community boards, and card-based filters, then calculates multiway equity using exact enumeration or Monte Carlo estimation. It separates **transport and request validation**, **session and range state**, and **domain algorithms** so each layer has a clear responsibility.

The database infrastructure defines the persistent schema and asynchronous connectivity independently of the current API runtime. Application sessions and computational state remain in memory.

This page is the **entry point for investigating the backend codebase**. It maps the components and common execution paths to their detailed documentation; it does not repeat their implementations or the client-facing API contracts.

## Documentation map

| If you want to understand… | Read |
| --- | --- |
| How FastAPI initializes resources and handles HTTP and SSE requests | [API architecture](api/README.md) |
| Which service owns sessions, player ranges, filters, and calculation jobs | [Services overview](services/README.md) |
| Database schema, asynchronous connectivity, migrations, and integration testing | [Database infrastructure](database.md) |
| Player slots, board transitions, per-street ranges, filter trees, and weighted coverage | [Range management](services/range-management.md) |
| Session expiry, equity job status, worker execution, progress, and cancellation | [Sessions and equity jobs](services/session-and-jobs.md) |
| Card IDs, board masks, `EncodedRange`, weighted ranges, and percentage-based ranges | [Card and range encoding](encoding.md) |
| Filter expression grammar, compilation, Boolean hand selections, and board-dependent filter definition generation | [Filter expression engine](filter-engine.md) |
| Five-card strength scores, ranking tables, and combinatorial lookup | [Five-card hand ranking](hand-ranking.md) |
| PLO showdown evaluation, exact equity, Monte Carlo samplers, and diagnostics | [Equity engine](equity.md) |

**For client integration**, use the [REST API contract](../api/rest.md) and [Equity SSE protocol](../api/sse.md). They specify HTTP routes, streamed JSON snapshots, validation, and client-visible behavior.

## System map

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

`EngineResources` is prepared at application startup and shared by session range managers. Mutable player, board, and filter state belongs to individual sessions. The equity HTTP router attaches jobs to sessions; the job factory itself does not manage session ownership.

The database infrastructure is currently separate from the application execution flow:

```text
Database infrastructure (db/)
├── models.py
│   ├── Base (shared declarative metadata)
│   ├── User (users table)
│   └── SavedRange (ranges table)
└── session.py
    ├── create_db_engine() → AsyncEngine
    └── create_session_factory() → async_sessionmaker

Schema management (alembic/)
├── env.py → synchronous migration engine
└── versions/ → versioned schema migrations

Database integration tests (tests/integration/)
├── disposable PostgreSQL container
├── automatic Alembic migrations
└── isolated asynchronous database sessions
```

PostgreSQL is not yet connected to the FastAPI application lifecycle. For schema definitions, connection management, and testing details, see [Database infrastructure](database.md).

## Follow an equity calculation

1. **Start a session.** The REST API creates a `Session`, which owns an initially empty `RangeManager`.
2. **Prepare the inputs.** REST calls populate player slots and set or clear the board. A newly constructed postflop street automatically receives filter definitions for occupied players; users can also create and activate filters. The services maintain each player's current weighted range for the active street.
3. **Submit work.** An HTTP `POST /equity` request validates the session and player configuration, attaches an `EquityJob`, and submits `RangeManager.calculate_equity()` to the application's thread pool.
4. **Compute equity.** `RangeManager` collects occupied players' current ranges. The equity engine validates and evaluates them with exact enumeration or Monte Carlo, using precomputed hand-ranking resources.
5. **Deliver updates.** The worker records progress on `EquityJob`. The SSE endpoint reads the latest snapshot and sends status/progress or the terminal result through an HTTP event stream.
6. **Disconnect or cancel.** The job belongs to the session, not the event stream. `EventSource` retries transient interruptions; a separate HTTP POST requests cooperative cancellation.

For the detailed request paths and lock boundaries, see [API architecture](api/README.md). For ownership and job lifecycles, see the [services overview](services/README.md).

## Key boundaries

- **Encoding versus ranking:** `encoder.py` converts card and range notation into compact representations. `rating.py` loads externally ordered *starting hands* for percentage ranges; `rank_combination.py` constructs *five-card showdown* ranking tables. These are different resources.
- **Filter definitions versus filter state:** The `filters` package evaluates expressions and generates board-dependent filter definitions. Each `PlayerRange` stores and evaluates these definitions against its base range, manages nested groups and activation, and derives weighted coverage and the current range.
- **Services versus computation:** `RangeManager` owns the shared PLO mode, street states, board transitions, and player slots. Each `PlayerRange` owns one player's filtering state for one street. Equity algorithms remain delegated to the `equity` package.
- **API versus services:** HTTP and SSE handlers validate requests, guard access to session state, and serialize results. They rely on services for state changes and on domain modules for evaluation.
- **Concurrency versus failure atomicity:** Clone-based range updates avoid publishing partially modified state after a failed operation. Session locks and REST mutation guards coordinate API-mediated access; they are not a general guarantee that every backend object is safe for arbitrary concurrent mutation.
- **Persistence versus application state:** The `db` package defines persistent user and saved range models. Application sessions, player ranges, filter states, and equity jobs remain process-local and are not persisted to PostgreSQL.
- **Database infrastructure versus business logic:** The database layer provides ORM models and asynchronous engine and session factories. User registration, authentication, range serialization, and saved range operations are not yet implemented.

## Where to investigate a problem

| Symptom or question | Start with |
| --- | --- |
| A request returns HTTP 404, 409, or 422 | [API architecture](api/README.md), then [REST API contract](../api/rest.md) |
| An SSE stream disconnects, reconnects, or reports unexpected status | [Equity SSE protocol](../api/sse.md), then [API architecture](api/README.md) |
| A session expires or a calculation does not stop immediately | [Sessions and equity jobs](services/session-and-jobs.md) |
| A board change regenerates filters, reuses cached filters, or changes the range size | [Range management](services/range-management.md) |
| A filter expression or generated definition matches unexpected hands | [Filter expression engine](filter-engine.md) |
| Filter activation or coverage is surprising | [Range management](services/range-management.md) |
| Percentage range selection looks wrong | [Card and range encoding](encoding.md) |
| Showdown strength or hand ordering looks wrong | [Five-card hand ranking](hand-ranking.md), then [Equity engine](equity.md) |
| Equity precision, sample counts, or execution time needs investigation | [Equity engine](equity.md) |
| Database connectivity, schema migration, or database integration test failure | [Database infrastructure](database.md) |

## Source directories

| Location | Primary role |
| --- | --- |
| [`src/plorangelab/api/`](../../src/plorangelab/api/) | FastAPI app, dependencies, schemas, and HTTP/SSE routers |
| [`src/plorangelab/services/`](../../src/plorangelab/services/) | Session lifecycle, range state, filtering state, and equity jobs |
| [`src/plorangelab/equity/`](../../src/plorangelab/equity/) | Equity calculation algorithms and results |
| [`src/plorangelab/filters/`](../../src/plorangelab/filters/) | Filter parsing, compilation, evaluation, and board-dependent definition generation |
| [`src/plorangelab/encoder.py`](../../src/plorangelab/encoder.py) and [`rating.py`](../../src/plorangelab/rating.py) | Card/range encoding and preflop starting-hand orderings |
| [`src/plorangelab/rank_combination.py`](../../src/plorangelab/rank_combination.py) and [`bootstrap.py`](../../src/plorangelab/bootstrap.py) | Five-card rank tables and shared engine resource initialization |
| [`src/plorangelab/db/`](../../src/plorangelab/db/) | ORM models, asynchronous database engine and session factories |
| [`alembic/`](../../alembic/) | Database schema migrations |
| [`tests/integration/`](../../tests/integration/) | PostgreSQL integration tests and disposable database fixtures |

The current API uses process-local sessions and an application-owned thread pool. Database infrastructure is implemented and independently tested, but is not yet integrated into the API runtime. See the subsystem documents for specific limitations and implementation guarantees; this overview is intentionally a map, not a detailed specification of individual components.
