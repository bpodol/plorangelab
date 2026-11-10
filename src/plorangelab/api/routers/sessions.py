from fastapi import APIRouter, HTTPException

from ..dependencies import SessionDep, SessionManagerDep, editable_session
from ..schemas import BoardRequest, BoardResponse, SessionResponse

router = APIRouter(
    prefix="/api/sessions",
    tags=["sessions"],
)


@router.post("", status_code=201, response_model=SessionResponse)
def create_session(session_manager: SessionManagerDep):
    session_id = session_manager.create_session()
    return SessionResponse(session_id=session_id)


@router.put("/{session_id}/board", response_model=BoardResponse)
def change_board(request: BoardRequest, session: SessionDep):
    with editable_session(session):
        try:
            session.manager.change_board(request.board)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        players = [
            len(player.current_range.hands) if player is not None else 0
            for player in session.manager.players
        ]

        return BoardResponse(players=players)
