import threading
from unittest.mock import AsyncMock

import pytest
from psycopg.errors import NotNullViolation, UniqueViolation
from sqlalchemy import event, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, SessionTransaction

from plorangelab.db import User, create_user
from plorangelab.security.passwords import verify_password
from plorangelab.services import users as user_service
from plorangelab.services.users import (
    UsernameAlreadyExistsError,
    register_user,
)


@pytest.mark.asyncio
async def test_register_user_persists_user_with_hashed_password(
    db_session: AsyncSession,
) -> None:
    username = "registered_user"
    password = "example_password"

    user = await register_user(db_session, username, password)

    assert user.id is not None
    assert user.username == username
    assert user.created_at is not None
    assert user.password_hash != password
    assert verify_password(password, user.password_hash) is True
    assert not db_session.in_transaction()

    user_id = user.id
    db_session.expunge_all()

    persisted_user = await db_session.get(User, user_id)

    assert persisted_user is not None
    assert persisted_user.username == username
    assert verify_password(password, persisted_user.password_hash) is True


@pytest.mark.asyncio
async def test_register_user_rejects_duplicate_username(
    db_session: AsyncSession,
) -> None:
    username = "duplicate_user"

    original_user = await register_user(db_session, username, "original_password")
    original_id = original_user.id
    original_hash = original_user.password_hash

    with pytest.raises(UsernameAlreadyExistsError) as exc_info:
        await register_user(db_session, username, "different_password")

    assert isinstance(exc_info.value.__cause__, IntegrityError)
    assert not db_session.in_transaction()

    result = await db_session.scalars(select(User).where(User.username == username))
    users = result.all()

    assert len(users) == 1
    assert users[0].id == original_id
    assert users[0].password_hash == original_hash
    assert verify_password("original_password", users[0].password_hash) is True


@pytest.mark.asyncio
async def test_register_user_session_remains_usable_after_duplicate(
    db_session: AsyncSession,
) -> None:
    await register_user(db_session, "first_user", "password")

    with pytest.raises(UsernameAlreadyExistsError):
        await register_user(db_session, "first_user", "password")

    second_user = await register_user(db_session, "second_user", "password")

    assert second_user.id is not None
    assert second_user.username == "second_user"
    assert not db_session.in_transaction()


@pytest.mark.asyncio
async def test_register_user_propagates_unrelated_integrity_error(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def invalid_create_user(
        session: AsyncSession,
        username: str,
        password_hash: str,
    ) -> User:
        await session.execute(
            insert(User).values(
                username=username,
                password_hash=None,
            )
        )
        raise AssertionError("Expected a NOT NULL violation")

    monkeypatch.setattr(user_service, "create_user", invalid_create_user)

    with pytest.raises(IntegrityError) as exc_info:
        await register_user(db_session, "new_user", "password")

    assert isinstance(exc_info.value.orig, NotNullViolation)
    assert not db_session.in_transaction()


@pytest.mark.asyncio
async def test_register_user_propagates_unrelated_unique_violation(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_user = await register_user(db_session, "existing_user", "password")
    existing_id = existing_user.id

    async def conflicting_create_user(
        session: AsyncSession,
        username: str,
        password_hash: str,
    ) -> User:
        await session.execute(
            insert(User).values(
                id=existing_id,
                username=username,
                password_hash=password_hash,
            )
        )
        raise AssertionError("Expected a primary key violation")

    monkeypatch.setattr(user_service, "create_user", conflicting_create_user)

    with pytest.raises(IntegrityError) as exc_info:
        await register_user(db_session, "new_user", "password")

    original_error = exc_info.value.orig

    assert isinstance(original_error, UniqueViolation)
    assert original_error.diag.constraint_name != "uq_users_username"
    assert not db_session.in_transaction()


@pytest.mark.asyncio
async def test_register_user_rolls_back_after_successful_flush(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("Unexpected registration failure")

    async def failing_create_user(
        session: AsyncSession,
        username: str,
        password_hash: str,
    ) -> User:
        await create_user(session, username, password_hash)
        raise error

    monkeypatch.setattr(user_service, "create_user", failing_create_user)

    with pytest.raises(RuntimeError) as exc_info:
        await register_user(db_session, "failed_user", "password")

    assert exc_info.value is error
    assert not db_session.in_transaction()

    user = await db_session.scalar(select(User).where(User.username == "failed_user"))

    assert user is None


@pytest.mark.asyncio
async def test_register_user_propagates_hashing_failure(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("Password hashing failed")

    def failing_hash_password(password: str) -> str:
        raise error

    create_user_mock = AsyncMock()

    monkeypatch.setattr(user_service, "hash_password", failing_hash_password)
    monkeypatch.setattr(user_service, "create_user", create_user_mock)

    with pytest.raises(RuntimeError) as exc_info:
        await register_user(db_session, "failed_user", "password")

    assert exc_info.value is error
    create_user_mock.assert_not_called()
    assert not db_session.in_transaction()


@pytest.mark.asyncio
async def test_register_user_hashes_password_before_transaction(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread_id = threading.get_ident()
    hashing_thread_ids: list[int] = []
    hashing_completed = threading.Event()
    transaction_start_states: list[bool] = []

    original_hash_password = user_service.hash_password

    def recording_hash_password(password: str) -> str:
        hashing_thread_ids.append(threading.get_ident())

        password_hash = original_hash_password(password)

        hashing_completed.set()
        return password_hash

    def record_transaction_start(
        session: Session,
        transaction: SessionTransaction,
    ) -> None:
        if transaction.parent is None:
            transaction_start_states.append(hashing_completed.is_set())

    monkeypatch.setattr(user_service, "hash_password", recording_hash_password)

    event.listen(
        db_session.sync_session,
        "after_transaction_create",
        record_transaction_start,
    )

    try:
        await register_user(db_session, "new_user", "password")
    finally:
        event.remove(
            db_session.sync_session,
            "after_transaction_create",
            record_transaction_start,
        )

    assert len(hashing_thread_ids) == 1
    assert hashing_thread_ids[0] != event_loop_thread_id
    assert hashing_completed.is_set()
    assert transaction_start_states == [True]


@pytest.mark.asyncio
async def test_register_user_rolls_back_on_commit_failure(
    db_session: AsyncSession,
) -> None:
    error = RuntimeError("Commit failed")

    def fail_commit(session: Session) -> None:
        raise error

    event.listen(
        db_session.sync_session,
        "before_commit",
        fail_commit,
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            await register_user(db_session, "failed_user", "password")
    finally:
        event.remove(
            db_session.sync_session,
            "before_commit",
            fail_commit,
        )

    assert exc_info.value is error
    assert not db_session.in_transaction()

    user = await db_session.scalar(select(User).where(User.username == "failed_user"))

    assert user is None
