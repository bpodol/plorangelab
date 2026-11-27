"""Database-backed behavioral tests for the user repository."""

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from plorangelab.db.repositories.users import (
    create_user,
    get_user_by_id,
    get_user_by_username,
)


@pytest.mark.asyncio
async def test_user_repository_lookups_retrieve_persisted_data(
    db_session: AsyncSession,
) -> None:
    user = await create_user(db_session, "repository_user", "hashed")
    user_id = user.id

    # Force subsequent lookups through SQL rather than the identity map.
    db_session.expunge_all()

    by_id = await get_user_by_id(db_session, user_id)
    assert by_id is not None
    assert by_id.id == user_id
    assert by_id.username == "repository_user"
    assert by_id.password_hash == "hashed"
    assert await get_user_by_id(db_session, uuid4()) is None

    by_username = await get_user_by_username(db_session, "repository_user")
    assert by_username is not None
    assert by_username.id == user_id
    assert by_username.username == "repository_user"
    assert by_username.password_hash == "hashed"
    assert await get_user_by_username(db_session, "missing_user") is None


@pytest.mark.asyncio
async def test_create_user_flushes_without_committing(
    db_session: AsyncSession,
) -> None:
    user = await create_user(db_session, "uncommitted_user", "hashed")

    assert user.id is not None
    assert user.username == "uncommitted_user"
    assert user.password_hash == "hashed"
    assert db_session.in_transaction()

    # Roll back the session transaction/savepoint. An internal commit would
    # end the transaction early and leave the row visible to this same session.
    await db_session.rollback()
    assert await get_user_by_username(db_session, "uncommitted_user") is None


@pytest.mark.asyncio
async def test_create_user_propagates_duplicate_username_error(
    db_session: AsyncSession,
) -> None:
    await create_user(db_session, "duplicate_repository_user", "first_hash")

    with pytest.raises(IntegrityError) as exc_info:
        await create_user(db_session, "duplicate_repository_user", "second_hash")

    assert exc_info.value.orig.sqlstate == "23505"
    await db_session.rollback()
