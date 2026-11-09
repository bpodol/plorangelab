
from fastapi import APIRouter, HTTPException

from ..dependencies import (
    PlayerId,
    SessionDep,
    editable_session,
)
from ..schemas import RangeRequest, RangeResponse

router = APIRouter(
    prefix="/api/sessions/{session_id}/players",
    tags=["players"],
)


@router.put("/{player_id}", response_model=RangeResponse)
def add_range(player_id: PlayerId, request: RangeRequest, session: SessionDep):
    with editable_session(session):
        try:
            session.manager.add_range(
                request.hands,
                request.mode,
                player_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        added_hands = (session.manager.players[player_id].current_range.hands)

        return RangeResponse(hands=len(added_hands))


@router.delete("/{player_id}", status_code=204)
def delete_range(player_id: PlayerId, session: SessionDep):
    with editable_session(session):
        try:
            session.manager.remove_range(player_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
