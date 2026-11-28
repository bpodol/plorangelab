"""Configuration, database dependency, and API resource lifecycle tests."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from plorangelab.api import dependencies
from plorangelab.api import main as api_module
from plorangelab.config import Settings


def test_settings_requires_non_empty_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_reads_and_freezes_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/test")

    settings = Settings()

    assert settings.database_url == "postgresql+psycopg://localhost/test"
    with pytest.raises(ValidationError):
        settings.database_url = "postgresql+psycopg://localhost/other"


@pytest.mark.asyncio
async def test_db_session_completes_and_is_independent_per_execution() -> None:
    sessions: list[object] = []
    closed: list[object] = []

    class SessionFactory:
        @asynccontextmanager
        async def __call__(self):
            session = object()
            sessions.append(session)
            try:
                yield session
            finally:
                closed.append(session)

    app = FastAPI()
    app.state.db_session_factory = SessionFactory()
    request = SimpleNamespace(app=app)
    yielded: list[object] = []

    for _ in range(2):
        dependency = dependencies.get_db_session(request)
        yielded.append(await anext(dependency))
        with pytest.raises(StopAsyncIteration):
            await anext(dependency)

    assert sessions[0] is yielded[0]
    assert sessions[1] is yielded[1]
    assert sessions[0] is not sessions[1]
    assert closed == sessions


@pytest.mark.asyncio
async def test_db_session_closes_and_propagates_request_error() -> None:
    session = object()
    events: list[str] = []

    @asynccontextmanager
    async def session_context():
        try:
            yield session
        finally:
            events.append("close")

    app = FastAPI()
    app.state.db_session_factory = Mock(return_value=session_context())
    dependency = dependencies.get_db_session(SimpleNamespace(app=app))
    await anext(dependency)

    with pytest.raises(RuntimeError, match="request failed"):
        await dependency.athrow(RuntimeError("request failed"))

    assert events == ["close"]


@pytest.mark.asyncio
async def test_lifespan_wires_resources_and_releases_them_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = object()
    engine = SimpleNamespace(dispose=AsyncMock())
    factory = object()
    manager = SimpleNamespace(clear=Mock())
    executor = SimpleNamespace(shutdown=Mock())
    order: list[str] = []
    engine.dispose.side_effect = lambda: order.append("dispose")
    manager.clear.side_effect = lambda: order.append("clear")

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/test")
    monkeypatch.setattr(api_module, "load_resources", Mock(return_value=resources))
    monkeypatch.setattr(api_module, "create_db_engine", Mock(return_value=engine))
    monkeypatch.setattr(
        api_module, "create_session_factory", Mock(return_value=factory)
    )
    monkeypatch.setattr(api_module, "SessionManager", Mock(return_value=manager))
    monkeypatch.setattr(api_module, "ThreadPoolExecutor", Mock(return_value=executor))

    async def fake_to_thread(function, *args, **kwargs):
        order.append("shutdown")
        return function(*args, **kwargs)

    monkeypatch.setattr(api_module.asyncio, "to_thread", fake_to_thread)
    app = FastAPI()

    async with api_module.lifespan(app):
        assert app.state.db_session_factory is factory
        assert app.state.session_manager is manager
        assert app.state.executor is executor
        api_module.create_db_engine.assert_called_once_with(
            "postgresql+psycopg://localhost/test"
        )
        api_module.create_session_factory.assert_called_once_with(engine)
        api_module.SessionManager.assert_called_once_with(resources)

    assert order == ["clear", "shutdown", "dispose"]
    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["session_factory", "executor", "manager"])
async def test_lifespan_cleans_up_resources_on_partial_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
) -> None:
    engine = SimpleNamespace(dispose=AsyncMock())
    executor = SimpleNamespace(shutdown=Mock())
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/test")
    monkeypatch.setattr(api_module, "load_resources", Mock(return_value=object()))
    monkeypatch.setattr(api_module, "create_db_engine", Mock(return_value=engine))
    monkeypatch.setattr(
        api_module,
        "create_session_factory",
        Mock(
            side_effect=RuntimeError("startup failed")
            if failure_at == "session_factory"
            else None,
            return_value=object(),
        ),
    )
    monkeypatch.setattr(
        api_module,
        "ThreadPoolExecutor",
        Mock(
            side_effect=RuntimeError("startup failed")
            if failure_at == "executor"
            else None,
            return_value=executor,
        ),
    )
    monkeypatch.setattr(
        api_module,
        "SessionManager",
        Mock(
            side_effect=RuntimeError("startup failed")
            if failure_at == "manager"
            else None,
            return_value=SimpleNamespace(clear=Mock()),
        ),
    )

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(api_module.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(RuntimeError, match="startup failed"):
        async with api_module.lifespan(FastAPI()):
            pytest.fail("lifespan should not yield after startup failure")

    engine.dispose.assert_awaited_once_with()
    if failure_at == "executor":
        executor.shutdown.assert_not_called()
    elif failure_at == "manager":
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_cleanup", ["manager", "executor"])
async def test_lifespan_continues_cleanup_after_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
    failing_cleanup: str,
) -> None:
    engine = SimpleNamespace(dispose=AsyncMock())
    manager = SimpleNamespace(clear=Mock())
    executor = SimpleNamespace(shutdown=Mock())
    if failing_cleanup == "manager":
        manager.clear.side_effect = RuntimeError("manager cleanup failed")
    else:
        executor.shutdown.side_effect = RuntimeError("executor cleanup failed")

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/test")
    monkeypatch.setattr(api_module, "load_resources", Mock(return_value=object()))
    monkeypatch.setattr(api_module, "create_db_engine", Mock(return_value=engine))
    monkeypatch.setattr(
        api_module,
        "create_session_factory",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(api_module, "SessionManager", Mock(return_value=manager))
    monkeypatch.setattr(api_module, "ThreadPoolExecutor", Mock(return_value=executor))

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(api_module.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(RuntimeError):
        async with api_module.lifespan(FastAPI()):
            pass

    engine.dispose.assert_awaited_once_with()
    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
