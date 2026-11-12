"""REST endpoints for session creation and community board management.

Session lifecycle operations are delegated to SessionManager,
while board changes are handled by the session's RangeManager.
"""

from fastapi import APIRouter, HTTPException

from ..dependencies import SessionDep, SessionManagerDep, editable_session
from ..schemas import BoardRequest, BoardResponse, SessionResponse

router = APIRouter(
    prefix="/api/sessions",
    tags=["sessions"],
)


@router.post("", status_code=201, response_model=SessionResponse)
def create_session(session_manager: SessionManagerDep):
    """Create a new session with six initially empty player slots.

    Returns:
        HTTP 201 with the generated session identifier.
    """

    session_id = session_manager.create_session()
    return SessionResponse(session_id=session_id)


@router.put("/{session_id}/board", response_model=BoardResponse)
def change_board(request: BoardRequest, session: SessionDep):
    """Update or clear the shared community board.

    Accepts a board containing 3, 4, or 5 cards in concatenated
    two-character notation. A null board clears the community
    cards and returns all players to preflop.

    Updates the ranges of occupied player slots and returns their
    resulting hand counts.

    Returns:
        HTTP 200 with six hand counts in player slot order.
        Empty slots are represented by zero.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the board change is invalid.
    """

    with editable_session(session):
        try:
            session.manager.change_board(request.board)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Return hand counts for all six slots, preserving their positions
        # so the frontend can map each count to the correct player.
        players = [
            len(player.current_range.hands) if player is not None else 0
            for player in session.manager.players
        ]

        return BoardResponse(players=players)
