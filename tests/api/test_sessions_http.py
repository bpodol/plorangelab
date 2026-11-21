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
        json={"hands": "AsAhQcJd,KsKhTd9c"},
    )
    assert added.status_code == 200
    assert added.json() == {"hands": 2}

    manager = client.app.state.session_manager.sessions[session_id].manager
    assert manager._current_state.ranges[3] is not None

    deleted = client.delete(url)
    assert deleted.status_code == 204
    assert manager._current_state.ranges[3] is None


def test_board_updates_and_clearing_persist(client: TestClient) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd,7s8s9cTc")
    manager = client.app.state.session_manager.sessions[session_id].manager
    url = f"/api/sessions/{session_id}/board"

    changed = client.put(url, json={"board": "AsKd2c"})
    assert changed.status_code == 200
    assert changed.json() == {
        "players": [
            {
                "base_hands": 1,
                "filtered_hands": 1,
                "filtered_coverage": 1.0,
                "filters": {},
            },
            None,
            None,
            None,
            None,
            None,
        ]
    }
    assert manager._current_state.board is not None
    assert len(manager._current_state.ranges[0].current_range.hands) == 1

    cleared = client.put(url, json={"board": None})
    assert cleared.status_code == 200
    assert cleared.json() == {
        "players": [
            {
                "base_hands": 2,
                "filtered_hands": 2,
                "filtered_coverage": 1.0,
                "filters": {},
            },
            None,
            None,
            None,
            None,
            None,
        ]
    }
    assert manager._current_state.board is None
    assert manager._active_street == 0
    assert len(manager._current_state.ranges[0].current_range.hands) == 2


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


def test_change_mode_succeeds_without_players(client: TestClient) -> None:
    session_id = create_session(client)
    manager = client.app.state.session_manager.sessions[session_id].manager

    assert manager.mode == 4

    response = client.put(
        f"/api/sessions/{session_id}/mode",
        json={"mode": 5},
    )

    assert response.status_code == 204
    assert response.content == b""
    assert manager.mode == 5
    assert manager._current_state.ranges == [None] * 6

    # New ranges must use the updated mode.
    add_player(client, session_id, 0, "AsAhKdQcJs")
    assert manager._current_state.ranges[0].base_range.hands.shape[1] == 5


def test_change_mode_rejects_existing_players(client: TestClient) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd")

    manager = client.app.state.session_manager.sessions[session_id].manager
    original_streets = manager._streets
    original_player = manager._current_state.ranges[0]

    response = client.put(
        f"/api/sessions/{session_id}/mode",
        json={"mode": 5},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Cannot change mode with active player ranges"}
    assert manager.mode == 4
    assert manager._streets is original_streets
    assert manager._current_state.ranges[0] is original_player


def test_change_mode_to_same_mode_is_no_op(client: TestClient) -> None:
    session_id = create_session(client)
    add_player(client, session_id, 0, "AsAhQcJd")

    manager = client.app.state.session_manager.sessions[session_id].manager
    original_streets = manager._streets

    response = client.put(
        f"/api/sessions/{session_id}/mode",
        json={"mode": 4},
    )

    assert response.status_code == 204
    assert manager.mode == 4
    assert manager._streets is original_streets


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": 0},
        {"mode": 6},
        {"mode": "invalid"},
        {},
    ],
)
def test_invalid_mode_returns_422(
    client: TestClient,
    payload: dict,
) -> None:
    session_id = create_session(client)
    manager = client.app.state.session_manager.sessions[session_id].manager

    response = client.put(
        f"/api/sessions/{session_id}/mode",
        json=payload,
    )

    assert response.status_code == 422
    assert manager.mode == 4


def test_change_mode_preserves_board_without_players(
    client: TestClient,
) -> None:
    session_id = create_session(client)
    manager = client.app.state.session_manager.sessions[session_id].manager

    board_response = client.put(
        f"/api/sessions/{session_id}/board",
        json={"board": "AsKd2c"},
    )
    assert board_response.status_code == 200

    original_state = manager._current_state
    original_board = original_state.board

    response = client.put(
        f"/api/sessions/{session_id}/mode",
        json={"mode": 5},
    )

    assert response.status_code == 204
    assert manager.mode == 5
    assert manager._current_state is original_state
    assert manager._current_state.board is original_board
