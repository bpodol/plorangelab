from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from plorangelab.api import main as api_module
from plorangelab.equity import SamplingExhaustedError
from plorangelab.services import range_manager as range_manager_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    resources = SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )
    load_resources = Mock(return_value=resources)
    monkeypatch.setattr(api_module, "load_resources", load_resources)

    # Using the context manager runs FastAPI's lifespan startup/shutdown.
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


def test_create_session(client: TestClient) -> None:
    first = create_session(client)
    second = create_session(client)

    assert first != second
    sessions = client.app.state.session_manager.sessions
    assert first in sessions
    assert second in sessions
    assert sessions[first].manager is not sessions[second].manager


def test_unknown_and_expired_sessions(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown = client.put("/api/sessions/not-a-session/board", json={"board": None})
    assert unknown.status_code == 404

    session_id = create_session(client)
    sessions = client.app.state.session_manager.sessions
    sessions[session_id].last_activity = 0.0
    monkeypatch.setattr(api_module, "monotonic", lambda: api_module.SESSION_TTL + 1)

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
    ("path_suffix", "payload"),
    [
        ("range", {"player_id": 6, "hands": "AsAhQcJd", "mode": 4}),
        ("range", {"player_id": 0, "hands": "AsAsQcJd", "mode": 4}),
        ("board", {"board": "As"}),
        ("equity", {"mode": 4, "target_moe": 0}),
    ],
)
def test_invalid_requests_return_422(
    client: TestClient,
    path_suffix: str,
    payload: dict,
) -> None:
    session_id = create_session(client)
    url = f"/api/sessions/{session_id}/{path_suffix}"
    request = client.put if path_suffix == "board" else client.post

    response = request(url, json=payload)
    assert response.status_code == 422


def test_equity_requires_two_players_and_matching_modes(client: TestClient) -> None:
    session_id = create_session(client)
    url = f"/api/sessions/{session_id}/equity"

    add_player(client, session_id, 0, "AsAhQcJd")
    one_player = client.post(url, json={"mode": 4, "target_moe": 0.002})
    assert one_player.status_code == 422

    add_player(client, session_id, 4, "KsKhTd9c8s", mode=5)
    wrong_mode = client.post(url, json={"mode": 4, "target_moe": 0.002})
    assert wrong_mode.status_code == 422


def test_calculate_equity_sparse_slots(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 4, "KsKhTd9c")
    add_player(client, session_id, 0, "AsAhQcJd")
    assert client.put(
        f"/api/sessions/{session_id}/board", json={"board": "2c3d4h"}
    ).status_code == 200

    result = SimpleNamespace(
        equities=[0.6, 0.4],
        method="exact",
        margins_of_error=None,
        simulations=None,
        attempts=None,
        stop_reason=None,
        sampler=None,
        effective_samples=None,
    )
    compute = Mock(return_value=result)
    monkeypatch.setattr(range_manager_module, "compute_equity", compute)

    response = client.post(
        f"/api/sessions/{session_id}/equity",
        json={"mode": 4, "target_moe": 0.002},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["equities"] == {"0": 0.6, "4": 0.4}
    assert body["details"]["method"] == "exact"

    compute.assert_called_once()
    kwargs = compute.call_args.kwargs
    assert kwargs["mode"] == 4
    assert kwargs["target_moe"] == pytest.approx(0.002)
    assert kwargs["board"].shape == (3,)

    manager = client.app.state.session_manager.sessions[session_id].manager
    assert kwargs["players"][0] is manager.players[0].active_street.ranges[-1]
    assert kwargs["players"][1] is manager.players[4].active_street.ranges[-1]
    assert kwargs["rank_lookup"] is client.app.state.session_manager.resources.rank_lookup


def test_sampling_exhaustion_is_handled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd")
    add_player(client, session_id, 1, "KsKhTd9c")
    monkeypatch.setattr(
        range_manager_module,
        "compute_equity",
        Mock(side_effect=SamplingExhaustedError("Sampling attempts exhausted")),
    )

    response = client.post(
        f"/api/sessions/{session_id}/equity",
        json={"mode": 4, "target_moe": 0.002},
    )
    assert response.status_code == 503  
    assert "Sampling attempts exhausted" in response.json()["detail"]


def test_unexpected_engine_error_is_not_converted_to_client_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd")
    add_player(client, session_id, 1, "KsKhTd9c")
    monkeypatch.setattr(
        range_manager_module,
        "compute_equity",
        Mock(side_effect=RuntimeError("internal failure")),
    )

    # TestClient raises server exceptions by default: an unexpected bug must
    # not be silently turned into an HTTP 422 response.
    with pytest.raises(RuntimeError, match="internal failure"):
        client.post(
            f"/api/sessions/{session_id}/equity",
            json={"mode": 4, "target_moe": 0.002},
        )
