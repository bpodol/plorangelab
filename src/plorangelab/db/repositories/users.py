from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from ..models import User


async def get_user_by_id(
    session: AsyncSession,
    user_id: UUID,
) -> User | None:
    user = await session.get(User, user_id)

    return user


async def get_user_by_username(
    session: AsyncSession,
    username: str,
) -> User | None:
    stmt = select(User).where(User.username == username)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    return user


async def create_user(
    session: AsyncSession,
    username: str,
    password_hash: str,
) -> User:
    user = User(username=username, password_hash=password_hash)
    session.add(user)
    await session.flush()

    return user