# Sessions and equity jobs

## Overview

The session and equity-job services manage the **lifetime of application state** and the **execution of equity calculations**. They do not implement hand-range filtering or equity algorithms themselves.

- **`SessionManager`** stores sessions in memory and handles creation, lookup, inactivity expiration, deletion, and cleanup.
- **`Session`** holds a session's `RangeManager`, optional `EquityJob`, last-activity timestamp, and lifecycle lock.
- **`EquityJob`** tracks a submitted calculation's future, status, cancellation signal, latest Monte Carlo progress, and a single subscriber.
- **`start_equity_job()`** submits a calculation through a caller-provided `ThreadPoolExecutor` and connects the worker to job progress and cancellation.

These services are connected through `Session.equity_job`, but **starting a job does not automatically attach it to a session**. The calling application is responsible for establishing that association and coordinating access to the session while doing so.

## Source files and ownership

| Source | Responsibility |
| --- | --- |
| [`session_manager.py`](../../../src/plorangelab/services/session_manager.py) | Session registry, lifetime rules, explicit deletion and cleanup. |
| [`equity_job.py`](../../../src/plorangelab/services/equity_job.py) | Worker submission, job state, cancellation, progress, subscriber notifications. |
| [`range_manager.py`](../../../src/plorangelab/services/range_manager.py) | Session-owned player/board/filter state and synchronous equity calculation entry point. |

```text
SessionManager
├── resources: EngineResources (shared)
└── sessions: dict[session_id, Session]
    └── Session
        ├── manager: RangeManager
        ├── equity_job: EquityJob | None
        │   ├── future: Future | None
        │   ├── cancel_event: threading.Event
        │   ├── progress: MonteCarloResult | None
        │   └── _subscriber: callback | None
        ├── last_activity: monotonic timestamp
        └── closed: bool

Caller-owned ThreadPoolExecutor
└── worker: RangeManager.calculate_equity(...)
    └── equity.compute_equity(...)
```

Each session gets a new `RangeManager`, but its manager is initialized with the shared `EngineResources` object. The job service receives an existing executor; neither `SessionManager` nor `start_equity_job()` constructs or shuts down that executor.

## Session lifecycle

### Creation and lookup

`SessionManager.create_session()` creates a random URL-safe identifier, constructs a `RangeManager` from shared engine resources, and records `time.monotonic()` as the initial activity timestamp. The identifier is used as the key in an in-memory session registry.

`get_session(session_id)` returns `None` for unknown, closed, or expired sessions. For a valid session it refreshes `last_activity` and returns the `Session` object itself. `Session.touch()` also refreshes the timestamp, but does nothing after the session has been closed.

The registry contains live Python objects, not serialized snapshots; callers receiving a `Session` have access to mutable state. Receiving a session reference does **not** prevent another request from closing that same session later.

### Inactivity expiration

`SESSION_TTL` is **30 minutes** (`30 * 60` seconds), measured using a monotonic clock. Expiration is evaluated **when `get_session()` is called**; the service has no periodic cleanup loop.

A session expires on lookup only when both conditions hold:

```text
no unfinished equity job
AND
current_monotonic_time - last_activity > SESSION_TTL
```

`Session.is_job_running()` treats a job as unfinished when it exists and either has no assigned future yet or its future has not completed. It does not use the job's user-facing status, and a cancellation request alone does not count as completion.

| On lookup | Behavior |
| --- | --- |
| Within inactivity timeout | Return the session and refresh activity. |
| Past timeout, no unfinished job | Mark closed, remove from registry, return `None`. |
| Past timeout, unfinished job | Return the session and refresh activity. |
| Past timeout, job already finished | Mark closed, remove from registry, return `None`. |

**Progress reports alone do not call `Session.touch()`** in either of these modules. A running job prevents expiration when a lookup happens; it does not continuously refresh the activity timestamp in the background.

### Deletion and cleanup

`delete_session(session_id)` is a no-op for an unknown identifier. Otherwise, it:

1. Looks up the session in the registry.
2. Acquires its lifecycle lock and marks it `closed`.
3. Requests cancellation if the associated equity job is unfinished.
4. Removes the registry entry if it still refers to the same `Session` instance.

Deletion **requests** cancellation via `EquityJob.cancel()`; it does not wait for a worker to stop, cancel its future directly, or shut down the thread pool. A worker may therefore briefly outlive its deleted session.

`SessionManager.clear()` takes a snapshot of current session IDs and deletes each using the same logic. Sessions created concurrently after that snapshot are not necessarily included.

The identity check before removal ensures cleanup does not delete a different object if the registry entry has changed between lookup and removal.

## Equity job lifecycle

### Submitting a calculation

`start_equity_job(manager, executor, mode, target_moe, on_finished=None)` creates an `EquityJob` and submits a worker function to the supplied `ThreadPoolExecutor`.

The worker calls:

```python
manager.calculate_equity(
    mode=mode,
    target_moe=target_moe,
    on_progress=on_progress,
    cancel_event=job.cancel_event,
)
```

`RangeManager.calculate_equity()` collects the current filtered ranges from occupied player slots, forwards the shared board and ranking resources, and returns `(player_indices, EquityResult)`. Details of range selection and equity algorithms belong to the [range-management](range-management.md) and [equity-engine](../equity.md) documents.

The worker receives the **existing `RangeManager` object**, not a snapshot. The job module itself does not lock or clone the manager. The surrounding application must ensure the manager is not changed unsafely during calculation.

The `on_finished` callback, if provided, is invoked in the worker's `finally` block when that worker exits, including when the calculation raises. It is not a general guarantee for jobs that never begin execution. The module does not define what the callback should do; any session bookkeeping belongs to its caller.

### Status model

`EquityJob.get_status()` derives status from `Future` state and, for a successful result, `EquityResult.stop_reason`:

| Status | Meaning |
| --- | --- |
| `pending` | No future assigned, or submitted but not yet running. |
| `running` | Future is executing. |
| `completed` | Future returned normally and result was not marked cancelled. |
| `cancelled` | Future was cancelled, or returned result has `stop_reason == "cancelled"`. |
| `failed` | Future finished with an exception. |

`completed` means **the worker returned normally**, not necessarily that the desired margin of error was achieved. The equity result's own stopping diagnostics explain why sampling stopped.

A newly created job may be pending while waiting for a free worker. `job.future` holds the `Future[tuple[list[int], EquityResult]]` returned by the executor.

### Cooperative cancellation

`EquityJob.cancel()` sets a `threading.Event`. It does **not** call `Future.cancel()`, interrupt the thread, or wait for completion. The event is passed into the range/equity calculation, which is responsible for checking it and stopping cooperatively.

Consequently, requesting cancellation does not immediately change a running job to `cancelled`. Until the future settles, its status may remain `pending` or `running`. A successful return with `stop_reason == "cancelled"` is displayed as cancelled; an exception is displayed as failed.

## Progress and subscriber notifications

During Monte Carlo execution, the equity calculation may call the supplied progress callback with a `MonteCarloResult`. `EquityJob.update_progress()`:

1. Replaces the saved `progress` snapshot while holding the job lock.
2. Releases that lock.
3. Notifies the current subscriber, if any.

`get_progress()` reads the latest snapshot under the same lock. **Progress is a latest-value snapshot, not an event history or queue**; intermediate reports can be superseded.

An equity job supports **one subscriber at a time**:

- `subscribe(callback)` replaces any previous subscriber. If the previous callback differs, it is invoked so its consumer can wake and notice the replacement.
- `unsubscribe(callback)` removes the subscriber only if it is the currently registered callback.
- `is_subscriber(callback)` checks callback identity.
- `notify_subscriber()` takes a snapshot of the callback under the lock and invokes it **outside** the lock.

Subscribing does not itself replay the latest state. A subscriber must read `get_progress()` and `get_status()` as needed. When the future finishes, a `Future.add_done_callback()` registration triggers a notification even if there were no progress updates.

Callbacks only signal that state may have changed; they do not receive progress or status as arguments. Their execution thread is not guaranteed to be the application's event-loop thread, so an asynchronous transport must marshal notifications appropriately.

## Synchronization boundaries

| Mechanism | What it coordinates | What it does **not** guarantee |
| --- | --- | --- |
| `SessionManager.lock` | Access to the session registry dictionary. | Protection for a session's range data. |
| `Session.lock` | Activity timestamp, closed flag, and coordinated lookup/deletion checks. | Automatic protection for operations on `RangeManager` or reassignment of `equity_job`. |
| `EquityJob.lock` | Progress snapshot and subscriber reference. | Exclusive access to the `RangeManager` being evaluated. |
| `threading.Event` | Cancellation request passed to equity calculation. | Immediate termination of a worker. |
| `Future` | Task lifecycle and final result or exception. | Isolation from concurrent changes to external objects. |

Registry operations release the manager lock before taking a session lock. This keeps the global registry lock from being held throughout session-specific work. The code does not implement one transaction across all these locks.

One subtlety: `Session.is_job_running()` reads `equity_job` and `future` without locking internally. The session-manager call sites use it while holding `Session.lock`, but other callers must coordinate accesses themselves.

## Typical application flow

The following is a **conceptual caller workflow**, not an API or WebSocket contract defined by these two files:

```text
1. SessionManager.create_session()
     → session ID and session-owned RangeManager

2. Caller obtains Session and configures ranges/board/filters
     → Session.manager

3. Caller starts calculation with a shared executor
     → start_equity_job(Session.manager, executor, ...)
     → EquityJob, which caller associates with Session.equity_job

4. Worker computes equity
     → progress callback → EquityJob.progress → subscriber notification
     → final (player_indices, EquityResult) or exception in Future

5. Caller reads progress/status/result
     → possibly streams state over its own transport

6. Session expires on later lookup, or caller explicitly deletes it
     → unfinished job receives cooperative cancellation request
```

This separation lets transport handlers manage how clients receive updates while the job layer remains independent of HTTP and WebSocket message formats.

## Boundaries and current constraints

- **In-memory only:** Sessions, jobs, progress snapshots, and subscriber references are not persisted in these services. Restarting the process loses them.
- **Lazy expiration:** Expired sessions are removed on lookup; the code does not sweep idle sessions automatically.
- **External job attachment:** `start_equity_job()` does not set `Session.equity_job`; that association and any replacement policy are caller responsibilities.
- **External executor lifecycle:** Thread-pool size, ownership, and shutdown are controlled elsewhere.
- **No automatic worker termination:** Session deletion requests cooperative cancellation and does not wait for completion.
- **One subscriber:** The job keeps only one callback and one current progress value, not a broadcast fan-out or durable event stream.
- **No automatic range-state synchronization:** Session and job locks do not make concurrent `RangeManager` mutations safe by themselves.
- **No transport specification:** REST routes, WebSocket events, and HTTP error mappings are implemented outside these files and should be documented at their own boundary.

## Related documentation

- [Services overview](README.md)
- [Range management](range-management.md)
- [Equity engine](../equity.md)
- [Encoding](../encoding.md)
