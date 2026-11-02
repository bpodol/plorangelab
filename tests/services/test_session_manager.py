"""Tests for the in-memory session manager."""

from types import SimpleNamespace

import pytest

from plorangelab.services import session_manager as session_module
from plorangelab.services.range_manager import RangeManager
from plorangelab.services.session_manager import SessionManager


@pytest.fixture
def resources() -> SimpleNamespace:
    
    return SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )


def test_create_session(
    resources: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "monotonic", lambda: 1234.5)
    manager = SessionManager(resources)

    session_id = manager.create_session()

    assert isinstance(session_id, str) and session_id
    assert len(manager.sessions) == 1

    session = manager.sessions[session_id]
    assert isinstance(session.manager, RangeManager)
    assert session.last_activity == 1234.5
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


def test_delete_and_clear_sessions(resources: SimpleNamespace) -> None:
    manager = SessionManager(resources)
    first_id = manager.create_session()
    second_id = manager.create_session()

    manager.delete_session(first_id)
    assert first_id not in manager.sessions
    assert second_id in manager.sessions

    manager.clear()
    assert manager.sessions == {}
