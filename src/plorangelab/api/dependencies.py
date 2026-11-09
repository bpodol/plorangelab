
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request

from plorangelab.services.session_manager import Session, SessionManager

PlayerId = Annotated[int, Path(ge=0, le=5)]


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


SessionManagerDep = Annotated[
    SessionManager,
    Depends(get_session_manager),
]


def get_session(
    session_id: str,
    session_manager: SessionManagerDep,
) -> Session:
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired",
        )

    return session


SessionDep = Annotated[Session, Depends(get_session)]


@contextmanager
def editable_session(session: Session) -> Iterator[None]:
    with session.lock:
        if session.closed:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired",
            )

        if session.is_job_running():
            raise HTTPException(
                status_code=409,
                detail="Cannot modify session during equity calculation",
            )

        yield
