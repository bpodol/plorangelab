"""HTTP/SSE tests for equity job submission, streaming, and cancellation."""

import asyncio
import json
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from plorangelab.api.routers import equity_sse as sse_module
from plorangelab.equity import SamplingExhaustedError
from plorangelab.services.equity_job import EquityJob
from plorangelab.services.session_manager import Session

from ._helpers import (
    add_player,
    add_two_players,
    create_session,
    equity_progress,
    equity_result,
)


def _url(session_id: str, job_id: str | None = None) -> str:
    base = f"/api/sessions/{session_id}/equity"
    return base if job_id is None else f"{base}/{job_id}"


def _attach_job(client: TestClient, session_id: str) -> tuple[Session, EquityJob]:
    """Install a controllable job in a real test session, without engine work."""
    session = client.app.state.session_manager.get_session(session_id)
    assert session is not None
    job = EquityJob()
    job.future = Future()
    session.equity_job = job
    return session, job


def _sse_data(body: str) -> dict:
    """Decode a single application event from an SSE response/chunk."""
    data_lines = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 1, body
    return json.loads(data_lines[0])


def test_submit_accepts_valid_players_and_registers_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    session = client.app.state.session_manager.get_session(session_id)
    assert session is not None

    job = EquityJob()
    job.future = Future()
    start = Mock(return_value=job)
    monkeypatch.setattr(sse_module, "start_equity_job", start)

    response = client.post(_url(session_id), json={"target_moe": 0.002})

    assert response.status_code == 202
    assert response.json() == {"status": "starting", "job_id": job.job_id}
    assert session.equity_job is job
    start.assert_called_once_with(
        manager=session.manager,
        executor=client.app.state.executor,
        target_moe=0.002,
        on_finished=session.touch,
    )


@pytest.mark.parametrize("player_count", [0, 1])
def test_submit_rejects_too_few_players(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, player_count: int
) -> None:
    session_id = create_session(client)
    if player_count:
        add_player(client, session_id, 0, "AsAhQcJd")
    start = Mock()
    monkeypatch.setattr(sse_module, "start_equity_job", start)

    response = client.post(_url(session_id), json={"target_moe": 0.002})

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid player count"
    start.assert_not_called()


@pytest.mark.parametrize("target_moe", [0, -0.001, 0.006])
def test_submit_validates_target_moe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, target_moe: float
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    start = Mock()
    monkeypatch.setattr(sse_module, "start_equity_job", start)

    response = client.post(_url(session_id), json={"target_moe": target_moe})

    assert response.status_code == 422
    start.assert_not_called()


def test_submit_rejects_second_unfinished_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    session, job = _attach_job(client, session_id)
    start = Mock()
    monkeypatch.setattr(sse_module, "start_equity_job", start)

    response = client.post(_url(session_id), json={"target_moe": 0.002})

    assert response.status_code == 409
    assert session.equity_job is job
    start.assert_not_called()


def test_cancel_requests_cooperative_cancellation(client: TestClient) -> None:
    session_id = create_session(client)
    _, job = _attach_job(client, session_id)

    response = client.post(f"{_url(session_id, job.job_id)}/cancel")

    assert response.status_code == 202
    assert response.json() == {"status": "cancelling", "job_id": job.job_id}
    assert job.cancel_event.is_set()
    assert job.future is not None and not job.future.done()


def test_cancel_rejects_unknown_and_finished_jobs(client: TestClient) -> None:
    session_id = create_session(client)
    assert client.post(f"{_url(session_id, 'missing')}/cancel").status_code == 404

    _, job = _attach_job(client, session_id)
    assert client.post(f"{_url(session_id, 'wrong-id')}/cancel").status_code == 404

    assert job.future is not None
    job.future.set_result(equity_result())
    response = client.post(f"{_url(session_id, job.job_id)}/cancel")
    assert response.status_code == 409
    assert not job.cancel_event.is_set()


def test_stream_rejects_unknown_job(client: TestClient) -> None:
    session_id = create_session(client)
    response = client.get(f"{_url(session_id, 'missing')}/events")
    assert response.status_code == 404


def test_stream_returns_completed_result_and_can_be_reconnected(
    client: TestClient,
) -> None:
    session_id = create_session(client)
    _, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_result(equity_result())

    url = f"{_url(session_id, job.job_id)}/events"
    first = client.get(url)
    second = client.get(url)

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")
    assert first.headers["cache-control"] == "no-cache, no-transform"
    assert first.headers["x-accel-buffering"] == "no"
    assert first.text == second.text

    event = _sse_data(first.text)
    assert event["job_id"] == job.job_id
    assert event["status"] == "completed"
    assert event["error"] is None
    assert event["details"]["equities"] == {"0": 0.6, "4": 0.4}
    assert event["details"]["method"] == "exact"
    assert event["details"]["margins_of_error"] is None


def test_stream_returns_cooperatively_cancelled_result(client: TestClient) -> None:
    session_id = create_session(client)
    _, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_result(replace(equity_progress(), stop_reason="cancelled"))

    response = client.get(f"{_url(session_id, job.job_id)}/events")
    event = _sse_data(response.text)

    assert response.status_code == 200
    assert event["status"] == "cancelled"
    assert event["error"] is None
    assert event["details"]["stop_reason"] == "cancelled"
    assert event["details"]["margins_of_error"] == {"0": 0.01, "4": 0.01}


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        (SamplingExhaustedError("No valid showdown"), "No valid showdown"),
        (RuntimeError("private engine error"), "Internal calculation error"),
    ],
)
def test_stream_reports_worker_exceptions(
    client: TestClient, exception: Exception, expected_error: str
) -> None:
    session_id = create_session(client)
    _, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_exception(exception)

    response = client.get(f"{_url(session_id, job.job_id)}/events")
    event = _sse_data(response.text)

    assert response.status_code == 200
    assert event["status"] == "failed"
    assert event["error"] == expected_error
    assert event["details"] is None


def test_stream_reports_cancelled_future(client: TestClient) -> None:
    session_id = create_session(client)
    _, job = _attach_job(client, session_id)
    assert job.future is not None and job.future.cancel()

    response = client.get(f"{_url(session_id, job.job_id)}/events")
    event = _sse_data(response.text)

    assert event["status"] == "cancelled"
    assert event["error"] == "Job had been cancelled"
    assert event["details"] is None


def test_stream_reports_invalid_final_result(client: TestClient) -> None:
    session_id = create_session(client)
    _, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_result(
        replace(equity_result(), method="unsupported")
    )  # Invalid schema field.

    response = client.get(f"{_url(session_id, job.job_id)}/events")
    event = _sse_data(response.text)

    assert event["status"] == "failed"
    assert event["error"] == "Internal server error"
    assert event["details"] is None


def test_stream_pushes_progress_and_then_one_final_event(client: TestClient) -> None:
    session_id = create_session(client)
    session, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_running_or_notify_cancel()

    async def exercise() -> None:
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        iterator = sse_module.stream_equity_events(
            job.job_id, session, request
        ).body_iterator
        try:
            first_chunk = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            assert not first_chunk.done()  # No progress yet: stream waits.

            # Mimic a progress callback originating in the worker thread.
            await asyncio.to_thread(job.update_progress, equity_progress())
            progress = _sse_data(await asyncio.wait_for(first_chunk, 1))
            assert progress["status"] == "running"
            assert progress["details"]["simulations"] == 1000
            assert progress["details"]["equities"] == {"0": 0.6, "4": 0.4}
            assert progress["details"]["margins_of_error"] == {"0": 0.01, "4": 0.01}

            next_chunk = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            job.notify_subscriber()  # Waking without a new snapshot must not resend it.
            await asyncio.sleep(0)
            assert not next_chunk.done()

            job.future.set_result(equity_result())
            job.notify_subscriber()
            completed = _sse_data(await asyncio.wait_for(next_chunk, 1))
            assert completed["status"] == "completed"
            assert completed["details"]["equities"] == {"0": 0.6, "4": 0.4}
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
        finally:
            await iterator.aclose()

    asyncio.run(exercise())
    assert job._subscriber is None


def test_reconnecting_replaces_previous_subscriber(client: TestClient) -> None:
    session_id = create_session(client)
    session, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_running_or_notify_cancel()

    async def exercise() -> None:
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        old = sse_module.stream_equity_events(
            job.job_id, session, request
        ).body_iterator
        new = sse_module.stream_equity_events(
            job.job_id, session, request
        ).body_iterator
        try:
            old_next = asyncio.create_task(anext(old))
            await asyncio.sleep(0)
            assert not old_next.done()

            new_next = asyncio.create_task(anext(new))
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(old_next, 1)

            await asyncio.to_thread(job.update_progress, equity_progress())
            event = _sse_data(await asyncio.wait_for(new_next, 1))
            assert event["status"] == "running"
            assert event["details"]["simulations"] == 1000

            # Cleanup of the displaced generator must not unregister the new one.
            assert job._subscriber is not None
        finally:
            await old.aclose()
            await new.aclose()

    asyncio.run(exercise())
    assert job._subscriber is None


def test_disconnection_stops_stream_without_cancelling_job(client: TestClient) -> None:
    session_id = create_session(client)
    session, job = _attach_job(client, session_id)

    async def exercise() -> None:
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        iterator = sse_module.stream_equity_events(
            job.job_id, session, request
        ).body_iterator
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        await iterator.aclose()

    asyncio.run(exercise())
    assert job._subscriber is None
    assert not job.cancel_event.is_set()


def test_stream_sends_heartbeat_without_progress(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    session, job = _attach_job(client, session_id)
    monkeypatch.setattr(sse_module, "HEARTBEAT_SECONDS", 0.01)

    async def exercise() -> None:
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        iterator = sse_module.stream_equity_events(
            job.job_id, session, request
        ).body_iterator
        try:
            assert await asyncio.wait_for(anext(iterator), 1) == ": heartbeat\n\n"
        finally:
            await iterator.aclose()

    asyncio.run(exercise())
    assert job._subscriber is None


def test_invalid_progress_is_skipped_without_ending_stream(client: TestClient) -> None:
    session_id = create_session(client)
    session, job = _attach_job(client, session_id)
    assert job.future is not None
    job.future.set_running_or_notify_cancel()

    async def exercise() -> None:
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        iterator = sse_module.stream_equity_events(
            job.job_id, session, request
        ).body_iterator
        try:
            next_chunk = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            invalid = replace(equity_progress(), method="invalid-method")
            await asyncio.to_thread(job.update_progress, invalid)
            await asyncio.sleep(0)
            assert not next_chunk.done()

            await asyncio.to_thread(job.update_progress, equity_progress())
            event = _sse_data(await asyncio.wait_for(next_chunk, 1))
            assert event["status"] == "running"
            assert event["details"]["method"] == "monte-carlo"
        finally:
            await iterator.aclose()

    asyncio.run(exercise())
