# Equity HTTP and SSE protocol

## Overview

Equity calculations are asynchronous **session jobs**. The client uses HTTP to submit or cancel a calculation and **Server-Sent Events (SSE)** to receive status, progress snapshots, and the terminal outcome. There is no WebSocket endpoint or `start` / `resume` / `cancel` JSON command protocol.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/sessions/{session_id}/equity` | Submit a new calculation. |
| `GET` | `/api/sessions/{session_id}/equity/{job_id}/events` | Receive snapshots over SSE. |
| `POST` | `/api/sessions/{session_id}/equity/{job_id}/cancel` | Request cooperative cancellation. |

Create the session and configure its ranges, board, and PLO mode through the [REST API](rest.md) before submitting a job. The engine uses the mode stored in the session; the equity submit request does **not** include a mode field.

The implementation lives in `src/plorangelab/api/routers/equity_sse.py` and the shared `EquityJob` service. See [backend API architecture](../backend/api/README.md) for locks, worker scheduling, and the notification bridge.

## Submit a calculation

```http
POST /api/sessions/{session_id}/equity
Content-Type: application/json
```

```json
{"target_moe": 0.0015}
```

Accepted submission returns **HTTP 202**:

```json
{
  "status": "starting",
  "job_id": "example-job-id"
}
```

`starting` acknowledges job submission, **not** that the worker is already running. The job may initially be `pending` in the shared executor queue. At least two player slots must have ranges on the active street; only one unfinished calculation is permitted per session. A conflicting submission is rejected rather than silently replacing the running job.

The job belongs to the **session**, not to any SSE connection. Its latest progress snapshot and future are held in memory. An application restart loses this state.

## Stream events

```http
GET /api/sessions/{session_id}/equity/{job_id}/events
Accept: text/event-stream
```

Successful subscriptions return **HTTP 200** and `Content-Type: text/event-stream`. The server sends default (unnamed) SSE messages, so browser clients can use `EventSource.onmessage`:

```text
data: {"job_id":"example-job-id","status":"running","error":null,"details":{"equities":{"0":0.62,"2":0.38},"method":"monte-carlo","margins_of_error":{"0":0.002,"2":0.002},"simulations":50000,"attempts":52000,"stop_reason":null,"sampler":"rejection","effective_samples":null}}

```

Each message ends with a **blank line**. `event.data` is a JSON string and must be parsed. A job with no new progress may produce an SSE comment heartbeat instead:

```text
: heartbeat

```

Heartbeats keep the stream active but are not delivered to `onmessage`.

### Event payload

Every application message has this outer shape:

```json
{
  "job_id": "example-job-id",
  "status": "running",
  "error": null,
  "details": null
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Identifies the submitted session job. |
| `status` | `pending` \| `running` \| `completed` \| `cancelled` \| `failed` | Current job status. |
| `error` | string or null | Error detail when the job fails. |
| `details` | object or null | Equity snapshot or terminal result, when available. |

When `details` is present, its fields are:

| Field | Type | Meaning |
| --- | --- | --- |
| `equities` | object mapping player ID to number | Equity per original zero-based player slot. |
| `method` | `exact` \| `monte-carlo` | Calculation method. |
| `margins_of_error` | object mapping player ID to number/null, or null | Per-player precision, when applicable. |
| `simulations` | integer or null | Number of completed Monte Carlo simulations. |
| `attempts` | integer or null | Sampling attempts. |
| `stop_reason` | `target_reached` \| `max_simulations` \| `cancelled` \| `attempt_limit` \| null | Why computation stopped, if available. |
| `sampler` | `rejection` \| `importance` \| null | Sampler selected for Monte Carlo. |
| `effective_samples` | number or null | Effective sample count, if applicable. |

The player IDs appear as **string keys in JSON objects** (for example, `"0"` and `"2"`), not as positions in a dense array. `details` can be `null` while the job is pending or before the first progress report. Fields that do not apply to exact calculation can also be null.

### Status and completion

| Status | Interpretation | Terminal? |
| --- | --- | --- |
| `pending` | Accepted but not yet executing. | No |
| `running` | Worker executing; may include the latest partial estimate. | No |
| `completed` | Worker returned normally; inspect `details.stop_reason` for stopping criteria. | Yes |
| `cancelled` | Worker ended due to cancellation (possibly returning partial details). | Yes |
| `failed` | Worker raised an error; inspect `error`. | Yes |

A terminal status is sent once and the **server ends that stream**. Browser `EventSource` clients should also call `.close()` on a terminal event, otherwise the browser can attempt to reconnect to the ended stream.

The stream publishes the **latest known snapshot** rather than maintaining a queue of every progress update. An exact calculation may finish without intermediate Monte Carlo progress. Do not assume every intermediate sample count is delivered.

## Request cancellation

```http
POST /api/sessions/{session_id}/equity/{job_id}/cancel
```

An accepted cancellation request returns **HTTP 202**:

```json
{
  "status": "cancelling",
  "job_id": "example-job-id"
}
```

Cancellation is **cooperative**: the endpoint signals a `threading.Event`, but does not interrupt a running thread or wait for it to finish. The client should keep reading SSE until a terminal event arrives. A worker may finish normally around the same time the cancellation request is accepted.

The active-job guard also applies to session mutations. Immediately submitting again after a cancellation acknowledgment may receive HTTP **409** until the prior job's future has finished.

## Disconnects, reconnection, and subscribers

- A transient network interruption puts native `EventSource` into its reconnection state. The browser retries the same GET stream without a custom WebSocket retry loop.
- The backend's `EquityJob` stores **one current subscriber**, not a broadcast fan-out. A newly subscribing stream replaces the earlier subscriber.
- On subscription, the stream inspects current job status/progress. It may skip intermediate progress that occurred while disconnected, and a terminal result can be returned from the stored job.
- Closing or losing an SSE stream does **not** inherently cancel the computation; cancellation requires the separate HTTP request.
- The current React hook deliberately treats an unrecoverable `EventSource.CLOSED` state or malformed message as a failure: it closes the stream, **requests cancellation**, releases the local job reference, and offers a fresh calculation rather than showing a manual Reconnect button.
- HTTP **202** confirms only that cancellation was requested. A quick new submission can be rejected while the old job is still ending.

The server does not provide a durable event log or cross-process resume mechanism. `EventSource` reconnection does not mean every past update is replayed.

## Failure and validation boundaries

- A missing or expired session cannot submit or retrieve its job.
- Submitting while a job is unfinished is rejected with HTTP **409**.
- Invalid request data (including bad target values) is rejected through FastAPI/Pydantic or domain validation.
- An HTTP error **before** a stream is established is distinct from a `failed` job event **inside** an established stream.
- The frontend parses `event.data` using JSON; its TypeScript interfaces do **not** provide runtime schema validation for all fields.

For specific request validation errors and other session endpoints, see the [REST API](rest.md). For the subscriber notification and worker lifecycle, see [Sessions and equity jobs](../backend/services/session-and-jobs.md).

## Related documentation

- [REST API](rest.md)
- [Backend API architecture](../backend/api/README.md)
- [Sessions and equity jobs](../backend/services/session-and-jobs.md)
- [Frontend state and data flow](../frontend/state-and-data-flow.md)
- [Equity engine](../backend/equity.md)
