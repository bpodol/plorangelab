# Backend services

## Start here

The services layer connects the application's **in-memory session lifecycle**, **player-range state**, **filtering**, **equity computation**, and **user registration**. It delegates domain algorithms to the encoder, filter-expression engine, and equity engine.

This page is a **navigation map**: use it to find which component owns a behavior, then follow the subsystem document or source file for details.

## Service map

```text
SessionManager (session_manager.py)
`-- Session
    |-- RangeManager (range_manager.py)
    |   |-- mode: PLO4 | PLO5
    |   |-- _active_street: Street
    |   `-- _streets: dict[Street, StreetState]
    |       `-- StreetState
    |           |-- board: Board | None
    |           `-- ranges[0..5]: PlayerRange | None
    |               `-- PlayerRange (player_range.py)
    `-- EquityJob | None (equity_job.py)
        |-- Future
        |-- cancellation event
        `-- latest progress + subscriber

EquityJob worker --calls--> RangeManager.calculate_equity()
                              `--> equity engine

register_user() (users.py)
|-- security/passwords.py (password hashing)
`-- db/repositories/users.py (user insertion)
    `-- PostgreSQL
```

Each session owns a `RangeManager`. The session may also reference a job created with `start_equity_job()`. The application caller attaches and coordinates that job; it is **not automatically registered by the job factory**.

The `ThreadPoolExecutor` used for equity jobs is supplied by the caller and is not owned by the service layer.

## Find the right component

| Investigating… | Start with | Source |
| --- | --- | --- |
| User registration, username normalization, duplicate detection | [User registration](#user-registration) | [`users.py`](../../../src/plorangelab/services/users.py) |
| Session creation, inactivity, deletion, cleanup | [Sessions and equity jobs](session-and-jobs.md) | [`session_manager.py`](../../../src/plorangelab/services/session_manager.py) |
| Job submission, status, cancellation, progress notifications | [Sessions and equity jobs](session-and-jobs.md) | [`equity_job.py`](../../../src/plorangelab/services/equity_job.py) |
| Adding/removing players, changing the shared board, invoking equity | [Range management](range-management.md) | [`range_manager.py`](../../../src/plorangelab/services/range_manager.py) |
| Flop/turn/river state reuse, range carryover, board blockers | [Range management](range-management.md) | [`range_manager.py`](../../../src/plorangelab/services/range_manager.py) |
| Generated filter installation, filter groups, activation, weighted coverage, selected ranges | [Range management](range-management.md) | [`player_range.py`](../../../src/plorangelab/services/player_range.py) |

## Follow a calculation through the code

A useful investigation path is:

1. **Find the session.** `SessionManager` returns a session with its own `RangeManager`.
2. **Prepare ranges.** `RangeManager` manages the shared PLO mode, all existing street states, and six player slots per street. Each occupied slot holds a `PlayerRange` responsible for that street's current filtered range. New postflop street states install automatically generated filter definitions without activating them.
3. **Start work.** The caller invokes `start_equity_job()` with a `RangeManager` and an existing executor. The worker calls `RangeManager.calculate_equity()`.
4. **Compute.** `calculate_equity()` gathers occupied players' current ranges and calls the equity engine.
5. **Report results.** `EquityJob` retains the latest progress snapshot and notifies one subscriber; its `Future` holds the final result or exception.
6. **Clean up.** `SessionManager` can expire a session on lookup or delete it explicitly, requesting cancellation for an unfinished associated job.

The sequence describes the **service-level flow**. Session-job association, API routing, SSE delivery, and executor creation are responsibilities of code outside the two job/session service modules.

## User registration

`services/users.py` coordinates account creation independently of HTTP. The caller supplies a validated username, plaintext password, and an `AsyncSession` without an active transaction.

1. Hash the password using `await asyncio.to_thread(hash_password, password)` before opening the session transaction. This uses the event loop's default executor, not the dedicated two-worker equity executor.
2. Enter `async with session.begin()`, convert the username to lowercase, and call the user repository's `create_user()` with the encoded hash. The repository adds and flushes the ORM object.
3. Exit the transaction context, committing on success, and return the persisted `User`. If an exception occurs inside the context, the transaction is rolled back.
4. Outside the transaction context, translate a SQLAlchemy `IntegrityError` only when its original Psycopg exception is `UniqueViolation` for `uq_users_username`. Raise `UsernameAlreadyExistsError` with the original exception as its cause. Other failures propagate.

PostgreSQL enforces username uniqueness. Lowercase storage makes registrations through this service case-insensitive; direct database or repository writes are not normalized. The API translates the business exception into HTTP 409.

[`security/passwords.py`](../../../src/plorangelab/security/passwords.py) uses a shared `argon2-cffi` `PasswordHasher` with its default Argon2id parameters. `hash_password()` generates a fresh random salt and returns the entire encoded hash, including algorithm parameters and salt. `verify_password()` returns `True` for a match and `False` for `VerifyMismatchError`; malformed hashes and unexpected verification failures propagate. Passwords are not stripped or case-normalized.

Registration does not authenticate a user, create an authentication session, or associate analysis sessions with an account. Login must apply the same username normalization. Registration rate limiting, an application-specific hashing concurrency limit, and deployment-time hashing benchmarks remain to be addressed before public deployment.

See [Database infrastructure](../database.md#request-scoped-sessions-and-transaction-ownership) for transaction ownership and [REST API](../../api/rest.md#register-user) for the HTTP contract.

## Subsystem documents

- **[Range management](range-management.md):** Six player slots, street transitions, filter-tree semantics, weighted coverage, and clone–modify–publish updates.
- **[Sessions and equity jobs](session-and-jobs.md):** Session expiration and deletion, worker submission, job states, progress/subscribers, cancellation, and synchronization boundaries.

## Domain algorithms used by services

- [Encoding](../encoding.md) — Parsing and storing weighted PLO ranges and board card representations.
- [Filter-expression engine](../filter-engine.md) — Parsing, compiling, and evaluating individual hand predicates, plus board-dependent definition generation.
- [Equity engine](../equity.md) — Exact and Monte Carlo equity calculations and stopping diagnostics.
- [Hand ranking](../hand-ranking.md) — Ranking-table generation and five-card ranking lookup.

## What belongs outside this folder?

Service documentation explains **state ownership and orchestration**, not the internals of domain algorithms or transport contracts. Document API routes, HTTP request/response formats, and SSE messages with their respective API components; keep the equity evaluator, encoder, and filter expression implementation details in the domain documents linked above.
