"""Coordinate equity jobs and asynchronous WebSocket updates.

Submits calculations to a shared thread pool, manages access to
session-owned jobs, and converts calculation results into WebSocket
messages.

Bridges worker-thread progress notifications to asyncio tasks,
allowing clients to receive updates or resume an existing job.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor
from typing import Any

from fastapi import WebSocketDisconnect

from plorangelab.equity import MonteCarloResult, SamplingExhaustedError
from plorangelab.services.equity_job import EquityJob, start_equity_job
from plorangelab.services.session_manager import Session

from .schemas import EquityDetails, EquityRequest, EquityResponse

logger = logging.getLogger(__name__)

SendMessage = Callable[[dict[str, Any]], Awaitable[None]]


async def submit_equity_calculation(
    session: Session,
    request: EquityRequest,
    executor: ThreadPoolExecutor,
) -> tuple[EquityJob | None, str | None]:
    """Validate and submit a new equity calculation for a session.

    Performs session checks and job submission in a worker thread
    to avoid blocking the asyncio event loop on threading locks.

    Under the session lock, verifies that the session is open,
    no previous calculation is unfinished, at least two players
    are present, and all player modes match the requested mode.

    On success, creates an EquityJob and attaches it to the session.
    The calculation uses the session's RangeManager, while its
    completion callback refreshes session activity.

    Args:
        session: Session whose ranges will be used for calculation.
        request: Validated equity calculation parameters.
        executor: Shared thread pool used for calculations.

    Returns:
        A tuple containing the new EquityJob and None on success,
        or None and an error message when submission is rejected.
    """

    def submit() -> tuple[EquityJob | None, str | None]:
        with session.lock:
            if session.closed:
                return None, "Session expired"

            if session.is_job_running():
                return None, "Calculation already running"

            players = [
                player for player in session.manager.players if player is not None
            ]
            if len(players) < 2 or any(
                player.mode != request.mode for player in players
            ):
                return None, "Invalid player count or equity mode"

            job = start_equity_job(
                manager=session.manager,
                executor=executor,
                mode=request.mode,
                target_moe=request.target_moe,
                on_finished=session.touch,
            )
            session.equity_job = job
            return job, None

    return await asyncio.to_thread(submit)


def get_equity_job(session: Session) -> EquityJob | None:
    """Retrieve the session's current equity job, if available.

    Returns None for closed sessions or sessions without a job.
    Completed jobs remain accessible for result resumption.

    Access to the job reference is synchronized by the session lock.
    """

    with session.lock:
        if session.closed:
            return None

        return session.equity_job


def build_final_message(job: EquityJob) -> dict:
    """Build a terminal WebSocket message from a finished equity job.

    Successful calculations produce a result message containing
    equities mapped to player slot indices and execution details.

    SamplingExhaustedError produces an error with its description.
    Cancelled futures produce a cancelled message. Unexpected
    exceptions are logged and represented by a generic error.

    Args:
        job: EquityJob whose future has reached a terminal state.

    Returns:
        JSON-serializable message containing a type and job_id,
        along with result data or error details.

    Notes:
        A calculation that cooperatively stops with
        stop_reason="cancelled" still produces a result message
        containing that stop reason.
    """

    try:
        ids, result = job.future.result()
        response = EquityResponse(
            equities=dict(zip(ids, result.equities, strict=True)),
            details=EquityDetails(
                method=result.method,
                margins_of_error=result.margins_of_error,
                simulations=result.simulations,
                attempts=result.attempts,
                stop_reason=result.stop_reason,
                sampler=result.sampler,
                effective_samples=result.effective_samples,
            ),
        )
        message = {
            "type": "result",
            "job_id": job.job_id,
            **response.model_dump(mode="json"),
        }

    except SamplingExhaustedError as exc:
        message = {
            "type": "error",
            "job_id": job.job_id,
            "detail": str(exc),
        }

    except CancelledError:
        return {
            "type": "cancelled",
            "job_id": job.job_id,
        }

    except Exception:
        logger.exception("Equity calculation failed")
        message = {
            "type": "error",
            "job_id": job.job_id,
            "detail": "Internal calculation error",
        }

    return message


async def publish_equity_updates(
    job: EquityJob,
    send: SendMessage,
) -> None:
    """Publish progress and terminal results for an equity job.

    Subscribes to job notifications using a callback that safely
    signals an asyncio.Event from worker threads.

    On each wake-up, checks job status and sends either the terminal
    result or the latest available progress snapshot.

    Only one subscriber is retained by EquityJob. If another
    connection replaces this subscriber, the current publisher exits.

    Args:
        job: EquityJob to monitor.
        send: Async callback responsible for sending WebSocket messages.

    Notes:
        Progress notifications are not queued. The publisher reads
        the most recent snapshot, so intermediate updates may be
        skipped when calculations progress faster than messages
        can be delivered.

        Disconnecting or stopping this publisher does not cancel
        the underlying equity calculation.

        The subscription is released when the publisher exits.
    """

    loop = asyncio.get_running_loop()
    changed = asyncio.Event()

    def wake() -> None:
        try:
            # Job callbacks may run on worker threads; schedule the event
            # update on the WebSocket's asyncio loop instead.
            loop.call_soon_threadsafe(changed.set)
        except RuntimeError:
            # The connection's event loop may have closed.
            pass

    job.subscribe(wake)
    last_progress: MonteCarloResult | None = None

    try:
        while True:
            changed.clear()

            # A newer connection may have replaced us.
            if not job.is_subscriber(wake):
                return

            status = job.get_status()

            # Terminal results take priority over pending progress snapshots.
            # Once the job finishes, send its final outcome and stop publishing.
            if status in ("completed", "cancelled", "failed"):
                if not job.is_subscriber(wake):
                    return

                await send(build_final_message(job))
                return

            progress = job.get_progress()

            if progress is not None and progress is not last_progress:
                if not job.is_subscriber(wake):
                    return

                last_progress = progress

                await send(
                    {
                        "type": "progress",
                        "job_id": job.job_id,
                        "simulations": progress.simulations,
                        "margins_of_error": progress.margins_of_error,
                    }
                )

            await changed.wait()

    except WebSocketDisconnect, OSError:
        pass

    finally:
        job.unsubscribe(wake)


def cancel_equity_calculation(session: Session) -> str | None:
    """Request cancellation of the session's unfinished equity job.

    Acquires the session lock and verifies that the session is open
    and has a job whose future has not completed.

    Cancellation is cooperative: the job's cancellation event is
    set, but the function does not wait for the worker to finish.

    Args:
        session: Session containing the calculation to cancel.

    Returns:
        None if cancellation was requested successfully, otherwise
        an error message describing why it was rejected.
    """

    with session.lock:
        if session.closed:
            return "Session expired"

        job = session.equity_job
        if job is None or job.future is None or job.future.done():
            return "No running calculation"

        job.cancel()
        return None
