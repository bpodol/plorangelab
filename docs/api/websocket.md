# Equity WebSocket protocol

## Overview

Equity calculations are started, monitored, resumed, and cancelled over a single WebSocket route:

```text
WS /api/sessions/{session_id}/equity/ws
```

The session must already exist, and its player ranges and board are configured through the [REST API](rest.md). The WebSocket sends and receives **JSON messages** with a `type` field. A job belongs to the **session**, not to a connection: disconnecting stops that connection's progress publisher but **does not cancel** its equity calculation.

This is the client protocol implemented by `routers/equity_ws.py` and `equity_ws_runtime.py`; see [backend API architecture](../backend/api/README.md) for implementation and synchronization details.

## Connection and session requirements

1. Create a session with `POST /api/sessions`.
2. Populate at least **two** player slots with ranges matching the desired PLO `mode` (4 or 5). Prepare the optional board and filters through REST.
3. Open `WS /api/sessions/{session_id}/equity/ws`.
4. Send `start`, or `resume` to attach to a previously submitted session job.

The router resolves the session before accepting the socket; if it is missing or expired, it closes with WebSocket code **1008**. It rechecks the **same session instance** and refreshes activity on every received command. If unavailable at that point, it closes the socket with code 1008.

There is no dedicated application-level ping/heartbeat message in these handlers. An idle open connection does not itself refresh the session's last-activity timestamp.

## Client commands

### `start` — submit a calculation

```json
{
  "type": "start",
  "mode": 4,
  "target_moe": 0.002
}
```

| Field | Required | Constraint |
| --- | --- | --- |
| `type` | Yes | `"start"`. |
| `mode` | Yes | `4` or `5`. |
| `target_moe` | No | Number **greater than 0 and at most 0.005**; defaults to **0.002** when omitted. |

A calculation is accepted only if the session is open, there is **no unfinished job**, at least two player slots are occupied, and **all occupied player modes match** the requested mode. Submission uses the application-owned equity executor.

**Immediate success acknowledgment:**

```json
{ "type": "started", "job_id": "example-job-id" }
```

`started` means the job was submitted. It may still be queued, running, or even finished by the time the client receives this message. The server then starts an update publisher for this connection.

**Error response:** `{"type": "error", "detail": "..."}` for invalid fields, invalid player configuration, an unfinished calculation, or a connection whose previous publisher is still delivering a result. Command-level errors generally have **no `job_id`**.

### `resume` — subscribe to the session's existing job

```json
{
  "type": "resume",
  "job_id": "example-job-id"
}
```

`job_id` is **optional**; if supplied, it must match the session's stored job ID. A completed job can be resumed so its terminal result is published again. The server stores only the session's **current** job reference, not a history of older jobs.

**Immediate acknowledgment:**

```json
{
  "type": "resumed",
  "job_id": "example-job-id",
  "status": "running"
}
```

`status` is one of `pending`, `running`, `completed`, `cancelled`, or `failed`, derived from the job's future and result. Following this acknowledgment, the publisher sends the latest available progress (if the job is still nonterminal) and/or a terminal message.

If there is no job or the specified ID differs, the server sends a command-level `error` message. Resuming from a **new connection** replaces the single subscriber previously registered with that job.

### `cancel` — request cancellation

```json
{ "type": "cancel" }
```

If the current job has an unfinished future, the server sets its cooperative cancellation event and acknowledges:

```json
{ "type": "cancel_requested" }
```

This acknowledgment **does not mean the worker has stopped**. The job must reach a terminal state separately. If the job is absent or already finished, the router returns a command-level `error` message with `"No running calculation"`.

`cancel` does not need a `job_id` and operates on the session's current job.

## Server-to-client messages

| `type` | When | Fields |
| --- | --- | --- |
| `started` | New job submitted. | `job_id` |
| `resumed` | Existing job selected. | `job_id`, `status` |
| `cancel_requested` | Cancellation signal set. | No additional fields |
| `progress` | New sampled Monte Carlo progress snapshot. | `job_id`, `simulations`, `margins_of_error` |
| `result` | Worker returned an equity result (including cooperative stops). | `job_id`, `equities`, `details` |
| `cancelled` | Job future raises `concurrent.futures.CancelledError`. | `job_id` |
| `error` | Command validation/rejection or job failure. | `detail`; `job_id` for job-result errors |

### Progress

**Illustrative progress event:**

```json
{
  "type": "progress",
  "job_id": "example-job-id",
  "simulations": 15000,
  "margins_of_error": [0.004, 0.004]
}
```

`simulations` is the number of completed samples represented by that update. `margins_of_error` is a per-participant array of optional numeric estimates. **The progress event does not include the latest equity estimates**, even though the underlying service result contains them.

Progress is a **snapshot, not a stream with guaranteed delivery of every intermediate update**. The job holds one latest progress value, and a publisher can skip older updates if computation or notification happens faster than socket delivery. Exact calculations may produce no Monte Carlo progress messages.

### Final result

**Illustrative result event:**

```json
{
  "type": "result",
  "job_id": "example-job-id",
  "equities": {
    "0": 0.55,
    "2": 0.45
  },
  "details": {
    "method": "monte-carlo",
    "margins_of_error": [0.003, 0.003],
    "simulations": 20000,
    "attempts": 21000,
    "stop_reason": "target_reached",
    "sampler": "rejection",
    "effective_samples": null
  }
}
```

The values above are examples, not a recorded calculation. `equities` includes **participating player slots only**, using their zero-based indices as keys. JSON object keys are strings even though the Python response schema uses `dict[int, float]`. Array diagnostics correspond to the participating players in slot order.

The `details` object always has these field names, though several values may be `null` depending on the calculation:

| Field | Type / allowed values | Meaning |
| --- | --- | --- |
| `method` | `"exact"` or `"monte-carlo"` | Calculation method selected by the equity engine. |
| `margins_of_error` | Array of number/null, or null | Per-participant estimated precision when available. |
| `simulations` | Integer or null | Completed samples when applicable. |
| `attempts` | Integer or null | Sampling attempts when applicable. |
| `stop_reason` | `"target_reached"`, `"max_simulations"`, `"cancelled"`, `"attempt_limit"`, or null | Why the calculation stopped. |
| `sampler` | `"rejection"`, `"importance"`, or null | Monte Carlo sampling strategy when applicable. |
| `effective_samples` | Number or null | Effective sample estimate when available. |

**Cancellation nuance:** the standard cooperative cancellation path can return a normal `EquityResult` with `details.stop_reason = "cancelled"`. In that case the server sends a **`result`** message, *not* a `cancelled` message. A distinct `cancelled` event is generated only when the future raises `CancelledError`.

### Errors

**Command-level error** (often without a `job_id`):

```json
{ "type": "error", "detail": "Calculation already running" }
```

Examples of other command errors from the router/runtime include `"No calculation to resume"`, `"Job ID does not match"`, `"No running calculation"`, `"Invalid player count or equity mode"`, `"Session expired"`, `"Previous result still being delivered"`, `"Expected a JSON object"`, and `"Unknown command"`. Invalid `start` fields return Pydantic validation error text in `detail`.

**Terminal job error:**

```json
{
  "type": "error",
  "job_id": "example-job-id",
  "detail": "Internal calculation error"
}
```

Unexpected calculation exceptions are logged server-side and reduced to the generic detail above. A `SamplingExhaustedError` instead exposes its exception message as `detail`.

**Malformed JSON syntax:** the router calls `websocket.receive_json()` without an explicit handler for JSON-decoding errors. Clients should not assume malformed JSON syntax always receives a structured `error` message. Valid JSON values that are not objects do receive `"Expected a JSON object"` after successful decoding.

## Typical client flow

```text
1. REST: create session and configure players/board/filters
2. Open WebSocket for that session
3. Send {"type":"start","mode":4,"target_moe":0.002}
4. Receive {"type":"started","job_id":"..."}
5. Receive zero or more {"type":"progress",...}
6. Receive a terminal {"type":"result" | "cancelled" | "error",...}
```

For reconnect/resume:

```text
1. WebSocket disconnects; the session's worker continues
2. Open a new WebSocket for the same session
3. Send {"type":"resume","job_id":"..."}  (job_id optional)
4. Receive {"type":"resumed","status":"...",...}
5. Receive latest progress if still running, then terminal result
```

Only the **most recently associated job** can be resumed, and its availability depends on the session still being open. A new `start` can replace the reference after the old job is finished.

### Practical client expectations

- Treat `started` and `cancel_requested` as **acknowledgments**, not terminal outcomes.
- Correlate progress and terminal events using `job_id`; a command-level error may omit that field.
- Treat `progress` as **best-effort latest state**; do not require every intermediate sample update.
- Continue to process `result` even after requesting cancellation, and inspect `details.stop_reason`.
- A newer subscribing connection displaces the previous publisher; the protocol does **not** broadcast to multiple listeners.
- The command handler and progress publisher serialize writes on the same socket, but they can observe independently changing job state. Do not assume an atomic exchange spanning both tasks.

## Related documentation

- [REST API contract](rest.md)
- [Backend API architecture](../backend/api/README.md)
- [Sessions and equity jobs](../backend/services/session-and-jobs.md)
- [Equity engine](../backend/equity.md)
