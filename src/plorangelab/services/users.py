import asyncio

from psycopg.errors import UniqueViolation
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from plorangelab.db import User, create_user
from plorangelab.security.passwords import hash_password


class UsernameAlreadyExistsError(Exception):
    """Raised when a username is already registered."""


async def register_user(session: AsyncSession, username: str, password: str) -> User:
    password_hash = await asyncio.to_thread(hash_password, password)
    username = username.lower()

    try:
        async with session.begin():
            user = await create_user(session, username, password_hash)

    except IntegrityError as exc:
        if (
            isinstance(exc.orig, UniqueViolation)
            and exc.orig.diag.constraint_name == "uq_users_username"
        ):
            raise UsernameAlreadyExistsError(username) from exc

        raise

    return user
