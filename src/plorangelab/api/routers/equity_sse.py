"""Submit equity jobs over HTTP and stream their results with SSE.

The submission endpoint schedules work in the shared thread pool and returns a
job ID immediately. A separate endpoint requests cooperative cancellation.
Clients connect to the job's event stream to receive the latest progress and a
single terminal result or error.

Execution and progress belong to EquityJob, not to the SSE connection: a client
may disconnect and reconnect without interrupting the calculation.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from concurrent.futures import CancelledError

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from plorangelab.equity import EquityResult, SamplingExhaustedError
from plorangelab.services.equity_job import EquityJob, start_equity_job
from plorangelab.services.session_manager import Session

from ..dependencies import SessionDep, locked_session
from ..schemas import (
    EquityDetailsSSE,
    EquityRequest,
    EquityResponseSSE,
    EquityStreamEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/sessions/{session_id}/equity",
    tags=["equity"],
)

TERMINAL_STATUSES = {"completed", "cancelled", "failed"}
HEARTBEAT_SECONDS = 15


def _get_job(session: Session, job_id: str) -> EquityJob:
    """Return the session's current job or raise HTTP 404.

    Sessions retain only their current job; older job IDs are not addressable.
    The lock protects the job lookup against concurrent replacement.
    """
    with locked_session(session):
        job = session.equity_job
        if job is None or job.job_id != job_id:
            raise HTTPException(404, "Equity job not found")
        return job


@router.post("", status_code=202, response_model=EquityResponseSSE)
def submit_equity(
    request: EquityRequest,
    session: SessionDep,
    http_request: Request,
) -> EquityResponseSSE:
    """Accept an equity calculation and return its ID without waiting.

    Under the session lock, reject overlapping jobs and sessions with fewer
    than two active players. The calculation continues in the shared executor;
    HTTP 202 means accepted, not completed. Results arrive through SSE.
    """
    with locked_session(session):
        if session.is_job_running():
            raise HTTPException(status_code=409, detail="Calculation already running")

        if session.manager.active_player_count < 2:
            raise HTTPException(status_code=422, detail="Invalid player count")

        job = start_equity_job(
            manager=session.manager,
            executor=http_request.app.state.executor,
            target_moe=request.target_moe,
            on_finished=session.touch,
        )
        session.equity_job = job

    return EquityResponseSSE(
        status="starting",
        job_id=job.job_id,
    )


@router.post("/{job_id}/cancel", status_code=202, response_model=EquityResponseSSE)
def cancel_equity(job_id: str, session: SessionDep) -> EquityResponseSSE:
    """Request cooperative cancellation of the session's current job.

    Setting the cancellation event does not stop the worker immediately. The
    stream reports the eventual terminal status when the worker has stopped.
    """
    with locked_session(session):
        job = session.equity_job
        if job is None or job.job_id != job_id:
            raise HTTPException(404, "Equity job not found")
        if not session.is_job_running():
            raise HTTPException(409, "No running calculation")
        job.cancel()

    return EquityResponseSSE(status="cancelling", job_id=job.job_id)


@router.get("/{job_id}/events")
def stream_equity_events(
    job_id: str,
    session: SessionDep,
    http_request: Request,
) -> StreamingResponse:
    """Stream a job's latest progress and eventual terminal outcome.

    The stream reads snapshots stored by EquityJob rather than queueing every
    update. Reconnecting replaces the previous subscriber and immediately reads
    the latest state; missed intermediate updates are not replayed. Disconnecting
    or replacing a stream does not cancel the calculation.

    Events use the default SSE message type (``data: <JSON>``), and the stream
    closes after emitting one terminal result or error.
    """
    job = _get_job(session, job_id)

    async def events() -> AsyncIterator[str]:
        """Bridge worker notifications to the asynchronous response stream."""
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        last_progress: EquityResult | None = None

        def wake() -> None:
            # Notifications can arrive outside the event-loop thread; asyncio
            # events must be set via the loop's thread-safe scheduler.
            try:
                loop.call_soon_threadsafe(changed.set)
            except RuntimeError:
                # The event loop may have closed after disconnection.
                pass

        # Only one stream subscribes at a time. A newer connection wakes the
        # displaced stream so it can notice the replacement and exit.
        job.subscribe(wake)

        try:
            while True:
                # Clear before reading job state: if an update arrives during
                # the read, its notification remains set for the next wait.
                changed.clear()

                if await http_request.is_disconnected():
                    return

                # A reconnect may have replaced this stream's callback.
                if not job.is_subscriber(wake):
                    return

                job_snapshot = job.get_snapshot()
                job_status = job_snapshot.status

                if job_snapshot.status in TERMINAL_STATUSES:
                    # Future.result() re-raises exceptions from the worker.
                    # Keep those separate from response validation errors.
                    try:
                        assert job.future is not None
                        result = job.future.result()

                    except SamplingExhaustedError as exc:
                        data = EquityStreamEvent(
                            job_id=job.job_id, status=job_status, error=str(exc)
                        )

                        yield f"data: {data.model_dump_json()}\n\n"
                        return

                    except CancelledError:
                        data = EquityStreamEvent(
                            job_id=job.job_id,
                            status=job_status,
                            error="Job had been cancelled",
                        )

                        yield f"data: {data.model_dump_json()}\n\n"
                        return

                    except Exception:
                        logger.exception("Equity job %s failed", job.job_id)

                        data = EquityStreamEvent(
                            job_id=job.job_id,
                            status=job_status,
                            error="Internal calculation error",
                        )

                        yield f"data: {data.model_dump_json()}\n\n"
                        return

                    # The calculation succeeded; validate the API payload
                    # independently so a schema error is not blamed on the engine.
                    try:
                        data = EquityStreamEvent(
                            job_id=job.job_id,
                            status=job_status,
                            details=EquityDetailsSSE.model_validate(result),
                        )

                        yield f"data: {data.model_dump_json()}\n\n"
                        return

                    except ValidationError:
                        logger.exception(
                            "Result model validation for job %s failed", job.job_id
                        )

                        data = EquityStreamEvent(
                            job_id=job.job_id,
                            status="failed",
                            error="Internal server error",
                        )

                        yield f"data: {data.model_dump_json()}\n\n"
                        return
                else:
                    progress = job_snapshot.progress

                    # EquityJob replaces progress objects on every update.
                    # Identity comparison prevents duplicate events on wakeups.
                    if progress is not None and progress is not last_progress:
                        last_progress = progress

                        try:
                            data = EquityStreamEvent(
                                job_id=job.job_id,
                                status=job_status,
                                details=EquityDetailsSSE.model_validate(
                                    job_snapshot.progress
                                ),
                            )

                        except ValidationError:
                            logger.exception(
                                "Progress model validation for job %s failed",
                                job.job_id,
                            )

                        else:
                            yield f"data: {data.model_dump_json()}\n\n"

                try:
                    await asyncio.wait_for(changed.wait(), timeout=HEARTBEAT_SECONDS)
                except TimeoutError:
                    # SSE comments keep idle connections alive without sending
                    # an application event to the frontend.
                    yield ": heartbeat\n\n"
        finally:
            # unsubscribe() checks callback identity, so an older stream
            # cannot remove the subscriber installed by a reconnect.
            job.unsubscribe(wake)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
