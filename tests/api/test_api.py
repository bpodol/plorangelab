"""HTTP and WebSocket API contracts for session-based range/equity operations."""

from collections.abc import Iterator
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from plorangelab.api import main as api_module
from plorangelab.equity import SamplingExhaustedError
from plorangelab.services import range_manager as range_manager_module
from plorangelab.services import session_manager as session_manager_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    resources = SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )
    load_resources = Mock(return_value=resources)
    monkeypatch.setattr(api_module, "load_resources", load_resources)

    # Runs the real FastAPI lifespan and its bounded ThreadPoolExecutor.
    with TestClient(api_module.app) as test_client:
        load_resources.assert_called_once_with()
        yield test_client


def create_session(client: TestClient) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def add_player(
    client: TestClient,
    session_id: str,
    slot: int,
    hands: str,
    mode: int = 4,
) -> None:
    response = client.post(
        f"/api/sessions/{session_id}/range",
        json={"player_id": slot, "hands": hands, "mode": mode},
    )
    assert response.status_code == 200, response.text


def add_two_players(client: TestClient, session_id: str) -> None:
    add_player(client, session_id, 0, "AsAhQcJd")
    add_player(client, session_id, 4, "KsKhTd9c")


def equity_result(equities: list[float] | None = None) -> SimpleNamespace:
    """Small completed result; ranking and sampling are tested elsewhere."""
    return SimpleNamespace(
        equities=equities if equities is not None else [0.6, 0.4],
        method="exact",
        margins_of_error=None,
        simulations=None,
        attempts=None,
        stop_reason=None,
        sampler=None,
        effective_samples=None,
    )


def ws_url(session_id: str) -> str:
    return f"/api/sessions/{session_id}/equity/ws"


def start_message(mode: int = 4, target_moe: float = 0.002) -> dict:
    return {"type": "start", "mode": mode, "target_moe": target_moe}


def test_create_session(client: TestClient) -> None:
    first = create_session(client)
    second = create_session(client)

    assert first != second
    sessions = client.app.state.session_manager.sessions
    assert first in sessions
    assert second in sessions
    assert sessions[first].manager is not sessions[second].manager


def test_unknown_and_expired_sessions(client: TestClient) -> None:
    unknown = client.put("/api/sessions/not-a-session/board", json={"board": None})
    assert unknown.status_code == 404

    session_id = create_session(client)
    sessions = client.app.state.session_manager.sessions
    sessions[session_id].last_activity = (
        monotonic() - session_manager_module.SESSION_TTL - 1
    )

    expired = client.put(f"/api/sessions/{session_id}/board", json={"board": None})
    assert expired.status_code == 404
    assert session_id not in sessions


def test_add_and_delete_range(client: TestClient) -> None:
    session_id = create_session(client)
    url = f"/api/sessions/{session_id}/range"

    added = client.post(
        url,
        json={"player_id": 3, "hands": "AsAhQcJd,KsKhTd9c", "mode": 4},
    )
    assert added.status_code == 200
    assert added.json() == {"hands": 2}

    manager = client.app.state.session_manager.sessions[session_id].manager
    assert manager.players[3] is not None

    deleted = client.delete(url, params={"player_id": 3})
    assert deleted.status_code == 204
    assert manager.players[3] is None


def test_board_updates_and_clearing_persist(client: TestClient) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd,7s8s9cTc")
    manager = client.app.state.session_manager.sessions[session_id].manager
    url = f"/api/sessions/{session_id}/board"

    changed = client.put(url, json={"board": "AsKd2c"})
    assert changed.status_code == 200
    assert manager.board is not None
    assert len(manager.players[0].active_street.ranges[-1].hands) == 1

    cleared = client.put(url, json={"board": None})
    assert cleared.status_code == 200
    assert manager.board is None
    assert manager.players[0].active_street is manager.players[0].preflop
    assert len(manager.players[0].active_street.ranges[-1].hands) == 2


@pytest.mark.parametrize(
    ("method", "suffix", "payload"),
    [
        ("post", "range", {"player_id": 6, "hands": "AsAhQcJd", "mode": 4}),
        ("post", "range", {"player_id": 0, "hands": "AsAsQcJd", "mode": 4}),
        ("put", "board", {"board": "As"}),
    ],
)
def test_invalid_http_input_returns_422(
    client: TestClient,
    method: str,
    suffix: str,
    payload: dict,
) -> None:
    session_id = create_session(client)
    response = getattr(client, method)(
        f"/api/sessions/{session_id}/{suffix}", json=payload
    )
    assert response.status_code == 422


def test_websocket_validates_equity_request(client: TestClient) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd")

    with client.websocket_connect(ws_url(session_id)) as ws:
        ws.send_json(start_message())
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "player count or equity mode" in message["detail"]

        add_player(client, session_id, 4, "KsKhTd9c8s", mode=5)
        ws.send_json(start_message())
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "player count or equity mode" in message["detail"]

        ws.send_json(start_message(target_moe=0))
        message = ws.receive_json()
        assert message["type"] == "error"
        assert "target_moe" in message["detail"]


def test_websocket_progress_and_sparse_slot_results(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 4, "KsKhTd9c")
    add_player(client, session_id, 0, "AsAhQcJd")
    assert client.put(
        f"/api/sessions/{session_id}/board", json={"board": "2c3d4h"}
    ).status_code == 200

    def fake_compute(**kwargs):
        kwargs["on_progress"](
            SimpleNamespace(simulations=1000, margins_of_error=[0.01, 0.01])
        )
        return equity_result()

    compute = Mock(side_effect=fake_compute)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)

    with client.websocket_connect(ws_url(session_id)) as ws:
        ws.send_json(start_message())
        assert ws.receive_json() == {"type": "started"}

        progress = ws.receive_json()
        assert progress["type"] == "progress"
        assert progress["simulations"] == 1000
        assert progress["margins_of_error"] == [0.01, 0.01]

        completed = ws.receive_json()
        assert completed["type"] == "result"
        assert completed["equities"] == {"0": 0.6, "4": 0.4}
        assert completed["details"]["method"] == "exact"

    compute.assert_called_once()
    kwargs = compute.call_args.kwargs
    assert kwargs["mode"] == 4
    assert kwargs["target_moe"] == pytest.approx(0.002)
    assert kwargs["board"].shape == (3,)

    session = client.app.state.session_manager.sessions[session_id]
    assert kwargs["players"][0] is session.manager.players[0].active_street.ranges[-1]
    assert kwargs["players"][1] is session.manager.players[4].active_street.ranges[-1]
    assert kwargs["rank_lookup"] is client.app.state.session_manager.resources.rank_lookup
    assert kwargs["cancel_event"] is session.equity_job.cancel_event


@pytest.mark.parametrize(
    ("failure", "expected_detail"),
    [
        (SamplingExhaustedError("Sampling attempts exhausted"), "Sampling attempts exhausted"),
        (RuntimeError("internal failure"), "Internal calculation error"),
    ],
)
def test_websocket_reports_calculation_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_detail: str,
) -> None:
    session_id = create_session(client)
    add_two_players(client, session_id)
    monkeypatch.setattr(
        range_manager_module,
        "compute_equity",
        Mock(side_effect=failure),
    )

    with client.websocket_connect(ws_url(session_id)) as ws:
        ws.send_json(start_message())
        assert ws.receive_json()["type"] == "started"
        message = ws.receive_json()
        assert message["type"] == "error"
        assert expected_detail in message["detail"]
        if isinstance(failure, RuntimeError) and not isinstance(
            failure, SamplingExhaustedError
        ):
            assert "internal failure" not in message["detail"]


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
