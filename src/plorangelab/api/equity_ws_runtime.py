"""Bridge synchronous equity jobs to asynchronous WebSocket updates."""

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
    with session.lock:
        if session.closed:
            return None

        return session.equity_job


def build_final_message(job: EquityJob) -> dict:
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
    loop = asyncio.get_running_loop()
    changed = asyncio.Event()

    def wake() -> None:
        try:
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
    with session.lock:
        if session.closed:
            return "Session expired"

        job = session.equity_job
        if job is None or job.future is None or job.future.done():
            return "No running calculation"

        job.cancel()
        return None
