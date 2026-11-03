"""Concurrent requests, cancellation and session lifecycle during WebSocket jobs."""

from threading import Event
from time import monotonic
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from plorangelab.services import range_manager as range_manager_module
from plorangelab.services import session_manager as session_manager_module

from ._helpers import (
    add_two_players,
    create_session,
    equity_result,
    start_message,
    ws_url,
)


def test_http_mutations_rejected_while_calculating(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    session = client.app.state.session_manager.sessions[session_id]
    release = Event()

    def fake_compute(**_kwargs):
        assert release.wait(timeout=10), "Calculation was never released"
        return equity_result()

    monkeypatch.setattr(range_manager_module, "compute_equity", fake_compute)
    base = f"/api/sessions/{session_id}"

    try:
        with client.websocket_connect(ws_url(session_id)) as ws:
            ws.send_json(start_message())
            assert ws.receive_json()["type"] == "started"

            # An active job prevents idle expiration even with an old timestamp.
            with session.lock:
                session.last_activity = monotonic() - session_manager_module.SESSION_TTL - 1

            assert client.put(f"{base}/board", json={"board": "2c3d4h"}).status_code == 409
            assert client.post(
                f"{base}/range",
                json={"player_id": 1, "hands": "QcQhJcTc", "mode": 4},
            ).status_code == 409
            assert client.delete(f"{base}/range", params={"player_id": 0}).status_code == 409

            assert session.manager.board is None
            assert session.manager.players[1] is None
            assert session.manager.players[0] is not None

            release.set()
            assert ws.receive_json()["type"] == "result"

        assert session.last_activity > monotonic() - 60
    finally:
        release.set()


def test_second_websocket_cannot_start_same_session_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    release = Event()
    entered = Event()

    def fake_compute(**_kwargs):
        entered.set()
        assert release.wait(timeout=10), "Calculation was never released"
        return equity_result()

    compute = Mock(side_effect=fake_compute)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)

    try:
        with (
            client.websocket_connect(ws_url(session_id)) as first,
            client.websocket_connect(ws_url(session_id)) as second,
        ):
            first.send_json(start_message())
            assert first.receive_json()["type"] == "started"
            assert entered.wait(timeout=5), "Worker did not start"

            second.send_json(start_message())
            rejected = second.receive_json()
            assert rejected["type"] == "error"
            assert "already running" in rejected["detail"]
            compute.assert_called_once()

            release.set()
            assert first.receive_json()["type"] == "result"
    finally:
        release.set()


def test_websocket_cancellation_and_completion_activity(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    session = client.app.state.session_manager.sessions[session_id]
    entered = Event()
    finish = Event()

    def fake_compute(**kwargs):
        entered.set()
        assert kwargs["cancel_event"].wait(timeout=10), "No cancellation received"
        assert finish.wait(timeout=10), "Cancellation test did not release worker"
        return equity_result()

    monkeypatch.setattr(range_manager_module, "compute_equity", fake_compute)

    try:
        with client.websocket_connect(ws_url(session_id)) as ws:
            ws.send_json(start_message())
            assert ws.receive_json()["type"] == "started"
            assert entered.wait(timeout=5)

            ws.send_json({"type": "cancel"})
            assert ws.receive_json() == {"type": "cancel_requested"}
            assert session.equity_job.cancel_event.is_set()

            with session.lock:
                session.last_activity = 0.0

            finish.set()
            assert ws.receive_json()["type"] == "result"

        assert session.last_activity > 0.0
    finally:
        finish.set()


def test_websocket_command_rejects_expired_session(client: TestClient) -> None:
    session_id = create_session(client)
    session = client.app.state.session_manager.sessions[session_id]

    with client.websocket_connect(ws_url(session_id)) as ws:
        with session.lock:
            session.last_activity = monotonic() - session_manager_module.SESSION_TTL - 1

        ws.send_json({"type": "cancel"})
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_json()
        assert exc_info.value.code == 1008

    assert session_id not in client.app.state.session_manager.sessions
