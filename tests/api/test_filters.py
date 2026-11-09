
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ._helpers import create_session


@pytest.fixture
def filter_context(client: TestClient):
    session_id = create_session(client)
    session = client.app.state.session_manager.get_session(session_id)
    assert session is not None
    base_url = f"/api/sessions/{session_id}/players/2"
    return base_url, session


def mock_operation(monkeypatch, session, name: str, *, result=None, error=None) -> Mock:
    operation = Mock(return_value=result, side_effect=error)
    monkeypatch.setattr(session.manager, name, operation)
    return operation


@pytest.mark.parametrize(
    ("payload", "name", "path"),
    [
        ({"expression": "top_pair"}, None, ()),
        (
            {"expression": "top_pair", "name": "Value", "path": ["Aggressive", "Flop"]},
            "Value",
            ("Aggressive", "Flop"),
        ),
    ],
)
def test_create_filter(client, filter_context, monkeypatch, payload, name, path):
    base_url, session = filter_context
    add_filter = mock_operation(monkeypatch, session, "add_filter", result=0.4)

    response = client.post(f"{base_url}/filters", json=payload)

    assert response.status_code == 201, response.text
    assert response.json() == {"coverage": pytest.approx(0.4)}
    add_filter.assert_called_once_with(
        expression="top_pair",
        player_index=2,
        name=name,
        path=path,
    )


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ({"name": "Draws"}, ()),
        ({"name": "Draws", "path": ["Aggressive", "Flop"]}, ("Aggressive", "Flop")),
    ],
)
def test_create_group(client, filter_context, monkeypatch, payload, path):
    base_url, session = filter_context
    add_group = mock_operation(monkeypatch, session, "add_group")

    response = client.post(f"{base_url}/groups", json=payload)

    assert response.status_code == 201, response.text
    assert response.json() == {"name": "Draws", "path": list(path)}
    add_group.assert_called_once_with(name="Draws", player_index=2, path=path)


@pytest.mark.parametrize("resource", ["filters", "groups"])
@pytest.mark.parametrize("active", [True, False])
def test_set_active(client, filter_context, monkeypatch, resource, active):
    base_url, session = filter_context
    set_active = mock_operation(monkeypatch, session, "set_filter_active", result=0.65)

    response = client.patch(
        f"{base_url}/{resource}/Draws",
        params=[("path", "Aggressive"), ("path", "Flop")],
        json={"active": active},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"coverage": pytest.approx(0.65)}
    set_active.assert_called_once_with(
        name="Draws",
        player_index=2,
        active=active,
        path=("Aggressive", "Flop"),
    )


@pytest.mark.parametrize("resource", ["filters", "groups"])
def test_delete_filter_or_group(client, filter_context, monkeypatch, resource):
    base_url, session = filter_context
    delete_filter = mock_operation(monkeypatch, session, "delete_filter", result=0.8)

    response = client.delete(
        f"{base_url}/{resource}/Draws",
        params=[("path", "Aggressive")],
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"coverage": pytest.approx(0.8)}
    delete_filter.assert_called_once_with("Draws", 2, ("Aggressive",))


def test_clear_filters_only_calls_clear(client, filter_context, monkeypatch):
    base_url, session = filter_context
    clear_filters = mock_operation(monkeypatch, session, "clear_filters", result=1.0)
    delete_filter = mock_operation(monkeypatch, session, "delete_filter")
    set_filter_active = mock_operation(monkeypatch, session, "set_filter_active")

    response = client.post(f"{base_url}/filters/clear")

    assert response.status_code == 200, response.text
    assert response.json() == {"coverage": pytest.approx(1.0)}
    clear_filters.assert_called_once_with(2)
    delete_filter.assert_not_called()
    set_filter_active.assert_not_called()


@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [
        ("POST", "filters", {"expression": ""}),
        ("POST", "groups", {"name": ""}),
        ("PATCH", "filters/Draws", {}),
        ("PATCH", "groups/Draws", {"active": "not-a-bool"}),
    ],
)
def test_invalid_request_body_returns_422(client, filter_context, method, suffix, body):
    base_url, _ = filter_context
    response = client.request(method, f"{base_url}/{suffix}", json=body)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("player_id", [-1, 6, "not-an-int"])
def test_invalid_player_id_returns_422(client, filter_context, player_id):
    base_url, _ = filter_context
    session_url = base_url.rsplit("/players/", 1)[0]
    response = client.post(
        f"{session_url}/players/{player_id}/filters",
        json={"expression": "top_pair"},
    )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    ("method", "suffix", "body", "operation_name"),
    [
        ("POST", "filters", {"expression": "invalid"}, "add_filter"),
        ("POST", "groups", {"name": "Draws"}, "add_group"),
        ("PATCH", "filters/Draws", {"active": True}, "set_filter_active"),
        ("PATCH", "groups/Draws", {"active": False}, "set_filter_active"),
        ("DELETE", "filters/Draws", None, "delete_filter"),
        ("DELETE", "groups/Draws", None, "delete_filter"),
        ("POST", "filters/clear", None, "clear_filters"),
    ],
)
def test_manager_value_error_returns_422(
    client, filter_context, monkeypatch, method, suffix, body, operation_name
):
    base_url, session = filter_context
    operation = mock_operation(
        monkeypatch, session, operation_name, error=ValueError("Invalid filter operation")
    )

    response = client.request(method, f"{base_url}/{suffix}", json=body)

    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "Invalid filter operation"}
    operation.assert_called_once()
