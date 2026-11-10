"""WebSocket equity protocol: validation, progress, errors, cancellation and concurrency."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from plorangelab.equity import SamplingExhaustedError
from plorangelab.services import range_manager as range_manager_module

from ._helpers import (
    add_player,
    add_two_players,
    create_session,
    equity_result,
    start_message,
    ws_url,
)


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
    assert (
        client.put(
            f"/api/sessions/{session_id}/board", json={"board": "2c3d4h"}
        ).status_code
        == 200
    )

    # Hold the worker after progress so the publisher cannot skip directly to
    # the terminal state. Progress events are intentionally coalesced.
    release = Event()

    def fake_compute(**kwargs):
        kwargs["on_progress"](
            SimpleNamespace(simulations=1000, margins_of_error=[0.01, 0.01])
        )
        assert release.wait(timeout=10), "Test did not release calculation"
        return equity_result()

    compute = Mock(side_effect=fake_compute)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)

    try:
        with client.websocket_connect(ws_url(session_id)) as ws:
            ws.send_json(start_message())
            started = ws.receive_json()
            assert started["type"] == "started"
            job_id = started["job_id"]
            assert isinstance(job_id, str) and job_id

            progress = ws.receive_json()
            assert progress["type"] == "progress"
            assert progress["job_id"] == job_id
            assert progress["simulations"] == 1000
            assert progress["margins_of_error"] == [0.01, 0.01]

            release.set()
            completed = ws.receive_json()
            assert completed["type"] == "result"
            assert completed["job_id"] == job_id
            assert completed["equities"] == {"0": 0.6, "4": 0.4}
            assert completed["details"]["method"] == "exact"
    finally:
        release.set()

    compute.assert_called_once()
    kwargs = compute.call_args.kwargs
    assert kwargs["mode"] == 4
    assert kwargs["target_moe"] == pytest.approx(0.002)
    assert kwargs["board"].shape == (3,)

    session = client.app.state.session_manager.sessions[session_id]
    assert kwargs["players"][0] is session.manager.players[0].current_range
    assert kwargs["players"][1] is session.manager.players[4].current_range
    assert (
        kwargs["rank_lookup"] is client.app.state.session_manager.resources.rank_lookup
    )
    assert kwargs["cancel_event"] is session.equity_job.cancel_event


@pytest.mark.parametrize(
    ("failure", "expected_detail"),
    [
        (
            SamplingExhaustedError("Sampling attempts exhausted"),
            "Sampling attempts exhausted",
        ),
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
