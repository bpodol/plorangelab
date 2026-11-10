"""HTTP API contracts for session, player range, and board operations."""

from time import monotonic

import pytest
from fastapi.testclient import TestClient

from plorangelab.services import session_manager as session_manager_module

from ._helpers import add_player, create_session


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
    url = f"/api/sessions/{session_id}/players/3"

    added = client.put(
        url,
        json={"hands": "AsAhQcJd,KsKhTd9c", "mode": 4},
    )
    assert added.status_code == 200
    assert added.json() == {"hands": 2}

    manager = client.app.state.session_manager.sessions[session_id].manager
    assert manager.players[3] is not None

    deleted = client.delete(url)
    assert deleted.status_code == 204
    assert manager.players[3] is None


def test_board_updates_and_clearing_persist(client: TestClient) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd,7s8s9cTc")
    manager = client.app.state.session_manager.sessions[session_id].manager
    url = f"/api/sessions/{session_id}/board"

    changed = client.put(url, json={"board": "AsKd2c"})
    assert changed.status_code == 200
    assert changed.json() == {"players": [1, 0, 0, 0, 0, 0]}
    assert manager.board is not None
    assert len(manager.players[0].current_range.hands) == 1

    cleared = client.put(url, json={"board": None})
    assert cleared.status_code == 200
    assert cleared.json() == {"players": [2, 0, 0, 0, 0, 0]}
    assert manager.board is None
    assert manager.players[0].active_street is manager.players[0].preflop
    assert len(manager.players[0].current_range.hands) == 2


@pytest.mark.parametrize(
    ("method", "player_id", "payload"),
    [
        ("put", 6, {"hands": "AsAhQcJd", "mode": 4}),
        ("put", 0, {"hands": "AsAsQcJd", "mode": 4}),
        ("put", "board", {"board": "As"}),
    ],
)
def test_invalid_http_input_returns_422(
    client: TestClient,
    method: str,
    player_id: str,
    payload: dict,
) -> None:
    session_id = create_session(client)
    response = getattr(client, method)(
        f"/api/sessions/{session_id}/players/{player_id}", json=payload
    )
    assert response.status_code == 422
