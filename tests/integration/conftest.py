import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from testcontainers.community.postgres import PostgresContainer

from plorangelab.db.models import User
from plorangelab.db.session import create_db_engine


def pytest_asyncio_loop_factories() -> dict[
    str, Callable[[], asyncio.AbstractEventLoop]
]:
    """Use an event loop compatible with asynchronous Psycopg."""
    return {
        "selector": asyncio.SelectorEventLoop,
    }


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Provide a disposable PostgreSQL instance for integration tests."""
    with PostgresContainer(
        image="postgres:18",
        driver="psycopg",
        username="test_user",
        password="test_password",
        dbname="plorangelab_test",
    ) as container:
        yield container


@pytest.fixture(scope="session")
def database_url(postgres_container: PostgresContainer) -> str:
    """Apply migrations and return the test database connection URL."""
    url = postgres_container.get_connection_url()

    project_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=project_root,
        env={**os.environ, "DATABASE_URL": url},
        check=True,
    )

    return url


@pytest_asyncio.fixture
async def db_engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """Provide an asynchronous database engine for a test."""
    engine = create_db_engine(database_url)

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(
    db_engine: AsyncEngine,
) -> AsyncIterator[AsyncSession]:
    """Provide an isolated database session for each test."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()

        try:
            async with AsyncSession(
                bind=connection,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            ) as session:
                yield session
        finally:
            await transaction.rollback()


@pytest_asyncio.fixture
async def db_user(db_session: AsyncSession) -> User:
    """Create a user within the isolated test transaction."""
    user = User(
        username="test_owner",
        password_hash="dummy_hash",
    )

    db_session.add(user)
    await db_session.flush()

    return user
