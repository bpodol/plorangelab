
"""WebSocket endpoint for live equity calculations."""

import asyncio
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from plorangelab.services.equity_job import EquityJob
from plorangelab.services.session_manager import Session, SessionManager

from ..equity_ws_runtime import (
    cancel_equity_calculation,
    get_equity_job,
    publish_equity_updates,
    submit_equity_calculation,
)
from ..schemas import EquityRequest

router = APIRouter()


@router.websocket("/api/sessions/{session_id}/equity/ws")
async def equity_websocket(websocket: WebSocket, session_id: str) -> None:
    session_manager: SessionManager = websocket.app.state.session_manager
    session: Session | None = await asyncio.to_thread(session_manager.get_session, session_id)

    if session is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    send_lock = asyncio.Lock()
    publisher_task: asyncio.Task[None] | None = None

    async def send(message: dict) -> None:
        async with send_lock:
            await websocket.send_json(message)

    async def replace_publisher(job: EquityJob) -> None:
        nonlocal publisher_task

        if publisher_task is not None:
            publisher_task.cancel()

            with suppress(asyncio.CancelledError):
                await publisher_task

        publisher_task = asyncio.create_task(
            publish_equity_updates(job, send)
        )

    try:
        while True:
            message = await websocket.receive_json()

            # Check expiration and refresh session activity.
            current_session = await asyncio.to_thread(
                session_manager.get_session, session_id
            )

            if current_session is not session:
                await websocket.close(code=1008)
                return

            if not isinstance(message, dict):
                await send({
                    "type": "error",
                    "detail": "Expected a JSON object",
                })
                continue

            command = message.get("type")

            if command == "start":
                try:
                    request = EquityRequest.model_validate({
                        "mode": message.get("mode"),
                        "target_moe": message.get("target_moe", 0.002),
                    })
                except ValidationError as exc:
                    await send({
                        "type": "error",
                        "detail": str(exc),
                    })
                    continue

                if publisher_task is not None and not publisher_task.done():
                    await send({
                        "type": "error",
                        "detail": "Previous result still being delivered",
                    })
                    continue

                job, error = await submit_equity_calculation(
                    session=session,
                    request=request,
                    executor=websocket.app.state.executor,
                )

                if error is not None:
                    await send({
                        "type": "error",
                        "detail": error,
                    })
                    continue

                assert job is not None

                await send({
                    "type": "started",
                    "job_id": job.job_id,
                })

                await replace_publisher(job)
                
            elif command == "resume":
                job = await asyncio.to_thread(
                    get_equity_job, session
                )

                if job is None:
                    await send({
                        "type": "error",
                        "detail": "No calculation to resume",
                    })
                    continue

                requested_id = message.get("job_id")

                if requested_id is not None and requested_id != job.job_id:
                    await send({
                        "type": "error",
                        "detail": "Job ID does not match",
                    })
                    continue

                await send({
                    "type": "resumed",
                    "job_id": job.job_id,
                    "status": job.get_status(),
                })

                await replace_publisher(job)

            elif command == "cancel":
                error = await asyncio.to_thread(
                    cancel_equity_calculation, session
                )

                if error is not None:
                    await send({
                        "type": "error",
                        "detail": error,
                    })
                else:
                    await send({"type": "cancel_requested"})

            else:
                await send({
                    "type": "error",
                    "detail": "Unknown command",
                })

    except WebSocketDisconnect:
        pass

    finally:
        if publisher_task is not None:
            publisher_task.cancel()

            with suppress(asyncio.CancelledError):
                await publisher_task
