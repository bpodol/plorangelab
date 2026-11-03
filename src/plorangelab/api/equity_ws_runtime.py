"""Bridge synchronous equity jobs to asynchronous WebSocket updates."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from plorangelab.equity import MonteCarloResult, SamplingExhaustedError
from plorangelab.services.equity_job import EquityJob, start_equity_job
from plorangelab.services.session_manager import Session

from .schemas import EquityDetails, EquityRequest, EquityResponse

logger = logging.getLogger(__name__)

ProgressQueue = asyncio.Queue[tuple[str, MonteCarloResult | None]]
SendMessage = Callable[[dict[str, Any]], Awaitable[None]]


async def submit_equity_calculation(
    session: Session,
    request: EquityRequest,
    executor: ThreadPoolExecutor,
) -> tuple[EquityJob | None, ProgressQueue, str | None]:
    loop = asyncio.get_running_loop()
    queue: ProgressQueue = asyncio.Queue(maxsize=2)

    def enqueue(event: tuple[str, MonteCarloResult | None]) -> None:
        # Keep the latest updates, including the final "done" notification.
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(event)

    def notify_progress(update: MonteCarloResult) -> None:
        # The engine calls this from a worker thread.
        loop.call_soon_threadsafe(enqueue, ("progress", update))

    def submit() -> tuple[EquityJob | None, str | None]:
        with session.lock:
            if session.closed:
                return None, "Session expired"

            if session.is_job_running():
                return None, "Calculation already running"

            players = [player for player in session.manager.players if player is not None]
            if len(players) < 2 or any(player.mode != request.mode for player in players):
                return None, "Invalid player count or equity mode"

            job = start_equity_job(
                manager=session.manager,
                executor=executor,
                mode=request.mode,
                target_moe=request.target_moe,
                notify=notify_progress,
                on_finished=session.touch,
            )
            session.equity_job = job
            return job, None

    job, error = await asyncio.to_thread(submit)

    if job is not None:
        assert job.future is not None
        job.future.add_done_callback(
            lambda _future: loop.call_soon_threadsafe(enqueue, ("done", None))
        )

    return job, queue, error


async def publish_equity_updates(
    job: EquityJob,
    queue: ProgressQueue,
    send: SendMessage,
) -> None:
    while True:
        event_type, update = await queue.get()

        if event_type == "progress":
            assert update is not None
            await send({
                "type": "progress",
                "simulations": update.simulations,
                "margins_of_error": update.margins_of_error,
            })
            continue

        # "done" is enqueued only after the Future completes.
        assert job.future is not None
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
            message = {"type": "result", **response.model_dump(mode="json")}

        except SamplingExhaustedError as exc:
            message = {"type": "error", "detail": str(exc)}

        except Exception:
            logger.exception("Equity calculation failed")
            message = {"type": "error", "detail": "Internal calculation error"}

        await send(message)
        return


def cancel_equity_calculation(session: Session) -> str | None:
    with session.lock:
        if session.closed:
            return "Session expired"

        job = session.equity_job
        if job is None or job.future is None or job.future.done():
            return "No running calculation"

        job.cancel()
        return None
