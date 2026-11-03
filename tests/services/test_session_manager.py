"""Tests for session creation, expiration, and job cleanup."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from plorangelab.services import session_manager as session_module
from plorangelab.services.equity_job import EquityJob
from plorangelab.services.range_manager import RangeManager
from plorangelab.services.session_manager import SESSION_TTL, SessionManager


@pytest.fixture
def resources() -> SimpleNamespace:
    return SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    time = {"now": 100.0}
    monkeypatch.setattr(session_module, "monotonic", lambda: time["now"])
    return time


def test_create_session(resources: SimpleNamespace, clock: dict[str, float]) -> None:
    manager = SessionManager(resources)

    session_id = manager.create_session()

    assert isinstance(session_id, str) and session_id
    assert len(manager.sessions) == 1

    session = manager.sessions[session_id]
    assert isinstance(session.manager, RangeManager)
    assert session.last_activity == clock["now"]
    assert session.manager.ratings is resources.ratings
    assert session.manager.rank_lookup is resources.rank_lookup


def test_sessions_have_independent_range_managers(resources: SimpleNamespace) -> None:
    manager = SessionManager(resources)

    first_id = manager.create_session()
    second_id = manager.create_session()

    assert first_id != second_id
    first = manager.sessions[first_id].manager
    second = manager.sessions[second_id].manager

    assert first is not second
    assert first.players is not second.players
    assert first.players == second.players == [None] * 6


def test_get_session_refreshes_activity(
    resources: SimpleNamespace,
    clock: dict[str, float],
) -> None:
    manager = SessionManager(resources)
    session_id = manager.create_session()
    session = manager.sessions[session_id]

    clock["now"] += SESSION_TTL - 1

    assert manager.get_session(session_id) is session
    assert session.last_activity == clock["now"]
    assert not session.closed


def test_idle_session_expires(
    resources: SimpleNamespace,
    clock: dict[str, float],
) -> None:
    manager = SessionManager(resources)
    session_id = manager.create_session()
    session = manager.sessions[session_id]

    clock["now"] += SESSION_TTL + 1

    assert manager.get_session(session_id) is None
    assert session.closed
    assert session_id not in manager.sessions
    assert manager.get_session("unknown-id") is None


def test_running_job_prevents_expiration(
    resources: SimpleNamespace,
    clock: dict[str, float],
) -> None:
    manager = SessionManager(resources)
    session_id = manager.create_session()
    session = manager.sessions[session_id]
    session.equity_job = EquityJob(future=Future())

    clock["now"] += SESSION_TTL + 1

    assert manager.get_session(session_id) is session
    assert session.last_activity == clock["now"]
    assert not session.closed


def test_touch_refreshes_activity_after_job_finishes(
    resources: SimpleNamespace,
    clock: dict[str, float],
) -> None:
    manager = SessionManager(resources)
    session_id = manager.create_session()
    session = manager.sessions[session_id]
    future: Future = Future()
    session.equity_job = EquityJob(future=future)

    clock["now"] += SESSION_TTL + 1
    future.set_result(None)
    session.touch()  # The job's on_finished callback calls this method.

    assert session.last_activity == clock["now"]
    assert manager.get_session(session_id) is session


def test_delete_and_clear_sessions(resources: SimpleNamespace) -> None:
    manager = SessionManager(resources)
    first_id = manager.create_session()
    second_id = manager.create_session()
    first = manager.sessions[first_id]
    second = manager.sessions[second_id]

    manager.delete_session(first_id)

    assert first.closed
    assert manager.get_session(first_id) is None
    assert second_id in manager.sessions
    assert not second.closed

    manager.clear()

    assert second.closed
    assert manager.sessions == {}


def test_delete_running_session_requests_cancellation(resources: SimpleNamespace) -> None:
    manager = SessionManager(resources)
    session_id = manager.create_session()
    session = manager.sessions[session_id]
    job = EquityJob(future=Future())
    session.equity_job = job

    manager.delete_session(session_id)

    assert session.closed
    assert job.cancel_event.is_set()
    assert manager.get_session(session_id) is None


def test_clear_cancels_all_running_jobs(resources: SimpleNamespace) -> None:
    manager = SessionManager(resources)
    session_ids = [manager.create_session() for _ in range(2)]
    sessions = [manager.sessions[session_id] for session_id in session_ids]
    jobs = [EquityJob(future=Future()) for _ in sessions]
    for session, job in zip(sessions, jobs, strict=True):
        session.equity_job = job

    manager.clear()

    assert manager.sessions == {}
    assert all(session.closed for session in sessions)
    assert all(job.cancel_event.is_set() for job in jobs)
