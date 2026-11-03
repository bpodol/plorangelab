"""WebSocket resume behavior after disconnect, completion, and failure."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from plorangelab.services import range_manager as range_manager_module

from ._helpers import (
    add_two_players,
    create_session,
    equity_result,
    start_message,
    ws_url,
)


def test_resume_without_job_reports_error(client: TestClient) -> None:
    session_id = create_session(client)
    with client.websocket_connect(ws_url(session_id)) as ws:
        ws.send_json({"type": "resume"})
        assert ws.receive_json() == {
            "type": "error",
            "detail": "No calculation to resume",
        }


def test_resume_rejects_incorrect_job_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    release = Event()

    def fake_compute(**_kwargs):
        assert release.wait(timeout=10), "Test did not release calculation"
        return equity_result()

    monkeypatch.setattr(range_manager_module, "compute_equity", fake_compute)
    try:
        with client.websocket_connect(ws_url(session_id)) as first:
            first.send_json(start_message())
            started = first.receive_json()
            assert started["type"] == "started"

        with client.websocket_connect(ws_url(session_id)) as second:
            second.send_json({"type": "resume", "job_id": "wrong-id"})
            message = second.receive_json()
            assert message["type"] == "error"
            assert "Job ID does not match" in message["detail"]
    finally:
        release.set()


def test_resume_running_job_receives_latest_progress_and_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    release = Event()
    entered = Event()

    def fake_compute(**kwargs):
        kwargs["on_progress"](
            SimpleNamespace(simulations=1000, margins_of_error=[0.01, 0.01])
        )
        entered.set()
        assert release.wait(timeout=10), "Test did not release calculation"
        return equity_result()

    compute = Mock(side_effect=fake_compute)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)
    try:
        with client.websocket_connect(ws_url(session_id)) as first:
            first.send_json(start_message())
            started = first.receive_json()
            assert started["type"] == "started"
            job_id = started["job_id"]
            assert entered.wait(timeout=5), "Worker did not report progress"
            # Old subscriber might send progress; we don't need to consume it.

        job = client.app.state.session_manager.sessions[session_id].equity_job
        assert job is not None and not job.future.done()

        with client.websocket_connect(ws_url(session_id)) as second:
            second.send_json({"type": "resume", "job_id": job_id})
            resumed = second.receive_json()
            assert resumed == {
                "type": "resumed",
                "job_id": job_id,
                "status": "running",
            }
            progress = second.receive_json()
            assert progress["type"] == "progress"
            assert progress["job_id"] == job_id
            assert progress["simulations"] == 1000

            release.set()
            completed = second.receive_json()
            assert completed["type"] == "result"
            assert completed["job_id"] == job_id
            assert completed["equities"] == {"0": 0.6, "4": 0.4}

        compute.assert_called_once()
    finally:
        release.set()


def test_resume_after_completion_returns_cached_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    release = Event()
    started_work = Event()

    def fake_compute(**_kwargs):
        started_work.set()
        assert release.wait(timeout=10), "Test did not release calculation"
        return equity_result()

    compute = Mock(side_effect=fake_compute)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)
    try:
        with client.websocket_connect(ws_url(session_id)) as first:
            first.send_json(start_message())
            started = first.receive_json()
            job_id = started["job_id"]
            assert started_work.wait(timeout=5)

        # Complete while there is NO WebSocket subscriber.
        release.set()
        job = client.app.state.session_manager.sessions[session_id].equity_job
        assert job is not None and job.future is not None
        job.future.result(timeout=5)

        with client.websocket_connect(ws_url(session_id)) as second:
            second.send_json({"type": "resume", "job_id": job_id})
            resumed = second.receive_json()
            assert resumed == {
                "type": "resumed",
                "job_id": job_id,
                "status": "completed",
            }
            result = second.receive_json()
            assert result["type"] == "result"
            assert result["job_id"] == job_id
            assert result["equities"] == {"0": 0.6, "4": 0.4}
        compute.assert_called_once()
    finally:
        release.set()


def test_resume_after_failure_returns_cached_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    release = Event()
    started_work = Event()

    def fake_compute(**_kwargs):
        started_work.set()
        assert release.wait(timeout=10), "Test did not release calculation"
        raise RuntimeError("sensitive engine details")

    compute = Mock(side_effect=fake_compute)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)
    try:
        with client.websocket_connect(ws_url(session_id)) as first:
            first.send_json(start_message())
            job_id = first.receive_json()["job_id"]
            assert started_work.wait(timeout=5)

        release.set()
        job = client.app.state.session_manager.sessions[session_id].equity_job
        assert job is not None and job.future is not None
        with pytest.raises(RuntimeError, match="sensitive engine details"):
            job.future.result(timeout=5)

        with client.websocket_connect(ws_url(session_id)) as second:
            second.send_json({"type": "resume", "job_id": job_id})
            assert second.receive_json() == {
                "type": "resumed",
                "job_id": job_id,
                "status": "failed",
            }
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["detail"] == "Internal calculation error"
            assert "sensitive engine details" not in str(error)
    finally:
        release.set()
