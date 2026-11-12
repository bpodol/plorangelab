# Backend services

## Start here

The services layer connects the application's **in-memory session lifecycle**, **player-range state**, **filtering**, and **equity computation**. It delegates domain algorithms to the encoder, filter-expression engine, and equity engine.

This page is a **navigation map**: use it to find which component owns a behavior, then follow the subsystem document or source file for details.

## Service map

```text
SessionManager                         (session_manager.py)
└── Session
    ├── RangeManager                  (range_manager.py)
    │   ├── board: Board
    │   └── PlayerRangeState[0..5]    (player_range_state.py)
    │       └── StreetState per street
    │           └── FilterManager     (filter_manager.py)
    └── EquityJob | None              (equity_job.py)
        ├── Future
        ├── cancellation event
        └── latest progress + subscriber

EquityJob worker ──calls──> RangeManager.calculate_equity()
                              └── equity engine
```

Each session owns a `RangeManager`. The session may also reference a job created with `start_equity_job()`. The application caller attaches and coordinates that job; it is **not automatically registered by the job factory**.

The `ThreadPoolExecutor` used for equity jobs is supplied by the caller and is not owned by any of these five service modules.

## Find the right component

| Investigating… | Start with | Source |
| --- | --- | --- |
| Session creation, inactivity, deletion, cleanup | [Sessions and equity jobs](session-and-jobs.md) | [`session_manager.py`](../../../src/plorangelab/services/session_manager.py) |
| Job submission, status, cancellation, progress notifications | [Sessions and equity jobs](session-and-jobs.md) | [`equity_job.py`](../../../src/plorangelab/services/equity_job.py) |
| Adding/removing players, changing the shared board, invoking equity | [Range management](range-management.md) | [`range_manager.py`](../../../src/plorangelab/services/range_manager.py) |
| Flop/turn/river state reuse, range carryover, board blockers | [Range management](range-management.md) | [`player_range_state.py`](../../../src/plorangelab/services/player_range_state.py) |
| Filter groups, activation, weighted coverage, selected ranges | [Range management](range-management.md) | [`filter_manager.py`](../../../src/plorangelab/services/filter_manager.py) |

## Follow a calculation through the code

A useful investigation path is:

1. **Find the session.** `SessionManager` returns a session with its own `RangeManager`.
2. **Prepare ranges.** `RangeManager` creates/replaces player slots and updates their shared board. Each `PlayerRangeState` determines the active street and its `FilterManager`'s current range.
3. **Start work.** The caller invokes `start_equity_job()` with a `RangeManager` and an existing executor. The worker calls `RangeManager.calculate_equity()`.
4. **Compute.** `calculate_equity()` gathers occupied players' current ranges and calls the equity engine.
5. **Report results.** `EquityJob` retains the latest progress snapshot and notifies one subscriber; its `Future` holds the final result or exception.
6. **Clean up.** `SessionManager` can expire a session on lookup or delete it explicitly, requesting cancellation for an unfinished associated job.

The sequence describes the **service-level flow**. Session-job association, API routing, WebSocket delivery, and executor creation are responsibilities of code outside the two job/session service modules.

## Subsystem documents

- **[Range management](range-management.md):** Six player slots, street transitions, filter-tree semantics, weighted coverage, and clone–modify–publish updates.
- **[Sessions and equity jobs](session-and-jobs.md):** Session expiration and deletion, worker submission, job states, progress/subscribers, cancellation, and synchronization boundaries.

## Domain algorithms used by services

- [Encoding](../encoding.md) — Parsing and storing weighted PLO ranges and board card representations.
- [Filter-expression engine](../filter-engine.md) — Parsing, compiling, and evaluating individual hand predicates.
- [Equity engine](../equity.md) — Exact and Monte Carlo equity calculations and stopping diagnostics.
- [Hand ranking](../hand-ranking.md) — Ranking-table generation and five-card ranking lookup.

## What belongs outside this folder?

Service documentation explains **state ownership and orchestration**, not the internals of domain algorithms or transport contracts. Document API routes, request/response formats, and WebSocket messages with their respective API components; keep the equity evaluator, encoder, and filter expression implementation details in the domain documents linked above.
