from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from plorangelab.db.models import SavedRange, User
from plorangelab.db.session import create_session_factory


def test_database_schema(database_url: str) -> None:
    engine = create_engine(database_url)

    try:
        with engine.connect() as connection:
            tables = inspect(connection).get_table_names()

            assert {"users", "ranges", "alembic_version"} <= set(tables)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_async_connection(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as connection:
        result = await connection.execute(text("SELECT current_database()"))

        assert result.scalar_one() == "plorangelab_test"


@pytest.mark.asyncio
async def test_session_factory(db_engine: AsyncEngine) -> None:
    session_factory = create_session_factory(db_engine)

    async with session_factory() as session:
        result = await session.execute(text("SELECT current_database()"))

        assert result.scalar_one() == "plorangelab_test"


@pytest.mark.asyncio
async def test_create_user(db_session: AsyncSession) -> None:
    user = User(
        username="test_user",
        password_hash="dummy_hash",
    )

    db_session.add(user)
    await db_session.commit()

    result = await db_session.execute(select(User).where(User.username == "test_user"))

    saved_user = result.scalar_one()

    assert saved_user.id is not None
    assert saved_user.username == "test_user"
    assert saved_user.created_at is not None


@pytest.mark.asyncio
async def test_transaction_isolation(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
) -> None:
    username = f"test_{uuid4().hex}"

    user = User(
        username=username,
        password_hash="dummy_hash",
    )

    db_session.add(user)
    await db_session.commit()

    query = select(User.id).where(User.username == username)

    # The record is visible within the test transaction.
    result = await db_session.execute(query)
    assert result.scalar_one() == user.id

    # The record is not visible through an independent connection.
    async with db_engine.connect() as connection:
        result = await connection.execute(query)
        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_duplicate_username(db_session: AsyncSession) -> None:
    db_session.add(User(username="alice", password_hash="dummy_hash"))
    await db_session.commit()

    db_session.add(User(username="alice", password_hash="another_hash"))

    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()

    assert exc_info.value.orig.sqlstate == "23505"

    await db_session.rollback()


@pytest.mark.asyncio
async def test_create_saved_range(
    db_session: AsyncSession,
    db_user: User,
) -> None:
    blob = b"\x00\x01\x02\xff"

    saved_range = SavedRange(
        user_id=db_user.id,
        name="BTN RFI",
        mode=5,
        range_blob=blob,
    )

    db_session.add(saved_range)
    await db_session.commit()

    range_id = saved_range.id

    db_session.expunge_all()

    restored = await db_session.get(SavedRange, range_id)

    assert restored is not None
    assert restored.id == range_id
    assert restored.user_id == db_user.id
    assert restored.name == "BTN RFI"
    assert restored.mode == 5
    assert restored.range_blob == blob
    assert restored.created_at is not None
    assert restored.updated_at is not None


@pytest.mark.asyncio
async def test_saved_range_invalid_user(
    db_session: AsyncSession,
) -> None:
    saved_range = SavedRange(
        user_id=uuid4(),
        name="BTN RFI",
        mode=5,
        range_blob=b"\x00\x01",
    )

    db_session.add(saved_range)

    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()

    assert exc_info.value.orig.sqlstate == "23503"

    await db_session.rollback()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [0, 3, 6])
async def test_saved_range_invalid_mode(
    db_session: AsyncSession,
    db_user: User,
    mode: int,
) -> None:
    saved_range = SavedRange(
        user_id=db_user.id,
        name="BTN RFI",
        mode=mode,
        range_blob=b"\x00\x01",
    )

    db_session.add(saved_range)

    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()

    assert exc_info.value.orig.sqlstate == "23514"
    assert exc_info.value.orig.diag.constraint_name == "ck_ranges_mode"

    await db_session.rollback()


@pytest.mark.asyncio
async def test_duplicate_saved_range(
    db_session: AsyncSession,
    db_user: User,
) -> None:
    db_session.add(
        SavedRange(
            user_id=db_user.id,
            name="BTN RFI",
            mode=5,
            range_blob=b"\x01",
        )
    )
    await db_session.commit()

    db_session.add(
        SavedRange(
            user_id=db_user.id,
            name="BTN RFI",
            mode=5,
            range_blob=b"\x02",
        )
    )

    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()

    assert exc_info.value.orig.sqlstate == "23505"
    assert exc_info.value.orig.diag.constraint_name == "uq_ranges_user_name_mode"

    await db_session.rollback()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "different_user, name, mode",
    [
        (False, "BTN RFI", 4),
        (False, "SB RFI", 5),
        (True, "BTN RFI", 5),
    ],
)
async def test_saved_range_valid_unique_combinations(
    db_session: AsyncSession,
    db_user: User,
    different_user: bool,
    name: str,
    mode: int,
) -> None:
    db_session.add(
        SavedRange(
            user_id=db_user.id,
            name="BTN RFI",
            mode=5,
            range_blob=b"\x01",
        )
    )
    await db_session.flush()

    owner_id = db_user.id

    if different_user:
        another_user = User(
            username="another_owner",
            password_hash="dummy_hash",
        )
        db_session.add(another_user)
        await db_session.flush()

        owner_id = another_user.id

    saved_range = SavedRange(
        user_id=owner_id,
        name=name,
        mode=mode,
        range_blob=b"\x02",
    )

    db_session.add(saved_range)
    await db_session.commit()

    assert saved_range.id is not None


@pytest.mark.asyncio
async def test_user_cascade_deletion(
    db_session: AsyncSession,
    db_user: User,
) -> None:
    another_user = User(
        username="another_owner",
        password_hash="dummy_hash",
    )
    db_session.add(another_user)
    await db_session.flush()

    db_session.add_all(
        [
            SavedRange(
                user_id=db_user.id,
                name="BTN RFI",
                mode=5,
                range_blob=b"\x01",
            ),
            SavedRange(
                user_id=db_user.id,
                name="SB RFI",
                mode=4,
                range_blob=b"\x02",
            ),
            SavedRange(
                user_id=another_user.id,
                name="BTN RFI",
                mode=5,
                range_blob=b"\x03",
            ),
        ]
    )
    await db_session.commit()

    deleted_user_id = db_user.id
    remaining_user_id = another_user.id

    await db_session.delete(db_user)
    await db_session.commit()

    db_session.expunge_all()

    assert await db_session.get(User, deleted_user_id) is None

    result = await db_session.execute(select(SavedRange))
    remaining_ranges = result.scalars().all()

    assert len(remaining_ranges) == 1
    assert remaining_ranges[0].user_id == remaining_user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["username", "password_hash"])
async def test_user_not_null(
    db_session: AsyncSession,
    field: str,
) -> None:
    user = User(
        username="alice",
        password_hash="dummy_hash",
    )

    setattr(user, field, None)
    db_session.add(user)

    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()

    assert exc_info.value.orig.sqlstate == "23502"
    assert exc_info.value.orig.diag.column_name == field

    await db_session.rollback()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ["user_id", "name", "mode", "range_blob"],
)
async def test_saved_range_not_null(
    db_session: AsyncSession,
    db_user: User,
    field: str,
) -> None:
    saved_range = SavedRange(
        user_id=db_user.id,
        name="BTN RFI",
        mode=5,
        range_blob=b"\x01",
    )

    setattr(saved_range, field, None)
    db_session.add(saved_range)

    with pytest.raises(IntegrityError) as exc_info:
        await db_session.commit()

    assert exc_info.value.orig.sqlstate == "23502"
    assert exc_info.value.orig.diag.column_name == field

    await db_session.rollback()
