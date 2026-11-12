"""Provide FastAPI dependencies and session access guards.

Resolves application sessions, validates player slot indices,
and provides context managers for synchronized session access.

Mutation guards prevent changes to session state while an
equity calculation is pending or running.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request

from plorangelab.services.session_manager import Session, SessionManager

PlayerId = Annotated[int, Path(ge=0, le=5)]


def get_session_manager(request: Request) -> SessionManager:
    """Retrieve the shared SessionManager from the FastAPI application state."""

    return request.app.state.session_manager


SessionManagerDep = Annotated[
    SessionManager,
    Depends(get_session_manager),
]


def get_session(
    session_id: str,
    session_manager: SessionManagerDep,
) -> Session:
    """Resolve an active session or raise HTTP 404.

    Retrieves the session through SessionManager, which checks
    expiration and refreshes its last activity timestamp.

    Args:
        session_id: Session identifier from the request path.
        session_manager: Application's shared session manager.

    Returns:
        Active Session instance.

    Raises:
        HTTPException: 404 if the session is missing or expired.
    """

    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired",
        )

    return session


SessionDep = Annotated[Session, Depends(get_session)]


@contextmanager
def locked_session(session: Session) -> Iterator[None]:
    """Hold a session's lock while accessing its mutable state.

    Rechecks whether the session has been closed after acquiring
    the lock, since deletion may occur after dependency resolution.

    Args:
        session: Session to protect.

    Raises:
        HTTPException: 404 if the session has been closed.

    Notes:
        The lock remains held throughout the caller's with block.
        This guard does not prohibit modifications while an equity
        calculation is running.
    """

    with session.lock:
        # The session may have been deleted after get_session()
        # returned, so validate it again while holding its lock.
        if session.closed:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired",
            )
        yield


@contextmanager
def editable_session(session: Session) -> Iterator[None]:
    """Allow session mutations only when no equity job is running.

    Acquires the session lock through locked_session(), verifies
    that the session is open, and rejects modifications while
    an equity calculation is pending or executing.

    The lock remains held throughout the caller's with block.

    Args:
        session: Session whose state will be modified.

    Raises:
        HTTPException: 404 if the session has been closed.
        HTTPException: 409 if an equity calculation is unfinished.
    """

    with locked_session(session):
        if session.is_job_running():
            raise HTTPException(
                status_code=409,
                detail="Cannot modify session during equity calculation",
            )
        yield
