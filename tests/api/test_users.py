from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from plorangelab.api.routers import users as users_router
from plorangelab.db import User
from plorangelab.services.users import UsernameAlreadyExistsError


@pytest.fixture
def register_user_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    mock = AsyncMock(
        return_value=User(
            username="test_user",
            password_hash="sensitive_password_hash",
        )
    )
    monkeypatch.setattr(users_router, "register_user", mock)
    return mock


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("Test_User123", "example_password"),
        ("Ab1", " Abc123 "),
        ("A" * 32, "x" * 128),
        ("unicode_user", "парольContraseña🔐"),
    ],
)
def test_register_user_returns_empty_created_response(
    client: TestClient,
    register_user_mock: AsyncMock,
    username: str,
    password: str,
) -> None:
    response = client.post(
        "/api/users",
        json={
            "username": username,
            "password": password,
        },
    )

    assert response.status_code == 201
    assert response.content == b""

    register_user_mock.assert_awaited_once()

    session, actual_username, actual_password = register_user_mock.call_args.args

    assert isinstance(session, AsyncSession)
    assert actual_username == username
    assert actual_password == password


def test_register_user_returns_conflict_for_existing_username(
    client: TestClient,
    register_user_mock: AsyncMock,
) -> None:
    register_user_mock.side_effect = UsernameAlreadyExistsError("existing_user")

    response = client.post(
        "/api/users",
        json={
            "username": "existing_user",
            "password": "example_password",
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Username already exists",
    }

    register_user_mock.assert_awaited_once()


@pytest.mark.parametrize(
    "username",
    [
        "",
        "ab",
        "a" * 33,
        "user name",
        " user",
        "user ",
        "user-name",
        "user.name",
        "user@name",
        "user\n",
        "пользователь",
        None,
        123,
        True,
        [],
        {},
    ],
)
def test_register_user_rejects_invalid_username(
    client: TestClient,
    register_user_mock: AsyncMock,
    username: object,
) -> None:
    response = client.post(
        "/api/users",
        json={
            "username": username,
            "password": "sensitive_password_123",
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(error["loc"] == ["body", "username"] for error in errors)
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)
    assert "sensitive_password_123" not in response.text

    register_user_mock.assert_not_called()


@pytest.mark.parametrize(
    "password",
    [
        "",
        "x" * 7,
        "x" * 129,
        None,
        12345678,
        True,
        [],
        {},
    ],
)
def test_register_user_rejects_invalid_password(
    client: TestClient,
    register_user_mock: AsyncMock,
    password: object,
) -> None:
    response = client.post(
        "/api/users",
        json={
            "username": "valid_user",
            "password": password,
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(error["loc"] == ["body", "password"] for error in errors)
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)

    register_user_mock.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "missing_fields"),
    [
        (
            {"username": "valid_user"},
            {"password"},
        ),
        (
            {"password": "sensitive_password_123"},
            {"username"},
        ),
        (
            {},
            {"username", "password"},
        ),
    ],
)
def test_register_user_rejects_missing_fields(
    client: TestClient,
    register_user_mock: AsyncMock,
    payload: dict[str, str],
    missing_fields: set[str],
) -> None:
    response = client.post("/api/users", json=payload)

    assert response.status_code == 422

    errors = response.json()["detail"]

    actual_missing_fields = {
        error["loc"][-1] for error in errors if error["type"] == "missing"
    }

    assert actual_missing_fields == missing_fields
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)
    assert "sensitive_password_123" not in response.text

    register_user_mock.assert_not_called()


@pytest.mark.parametrize(
    "password",
    [
        "pW!42",
        "sensitive_password_" * 10,
    ],
)
def test_registration_validation_errors_do_not_expose_password(
    client: TestClient,
    register_user_mock: AsyncMock,
    password: str,
) -> None:
    response = client.post(
        "/api/users",
        json={
            "username": "valid_user",
            "password": password,
        },
    )

    assert response.status_code == 422
    assert password not in response.text

    errors = response.json()["detail"]

    assert errors
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)

    register_user_mock.assert_not_called()


def test_register_user_rejects_invalid_body_without_exposing_input(
    client: TestClient,
    register_user_mock: AsyncMock,
) -> None:
    password = "sensitive_password_123"

    response = client.post(
        "/api/users",
        json=[
            {
                "username": "valid_user",
                "password": password,
            }
        ],
    )

    assert response.status_code == 422
    assert password not in response.text

    errors = response.json()["detail"]

    assert errors
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)

    register_user_mock.assert_not_called()


def test_register_user_rejects_malformed_json(
    client: TestClient,
    register_user_mock: AsyncMock,
) -> None:
    password = "sensitive_password_123"

    response = client.post(
        "/api/users",
        content=(f'{{"username": "valid_user", "password": "{password}"'),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert password not in response.text

    errors = response.json()["detail"]

    assert any(error["type"] == "json_invalid" for error in errors)
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)

    register_user_mock.assert_not_called()


def test_register_user_propagates_unexpected_service_error(
    client: TestClient,
    register_user_mock: AsyncMock,
) -> None:
    error = RuntimeError("Unexpected registration failure")
    register_user_mock.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        client.post(
            "/api/users",
            json={
                "username": "valid_user",
                "password": "example_password",
            },
        )

    assert exc_info.value is error

    register_user_mock.assert_awaited_once()
