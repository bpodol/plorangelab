"""REST endpoints for creating, replacing, and removing player ranges.

Each session supports six player slots, indexed from 0 to 5.
Range encoding and player state management are delegated to
the session's RangeManager.
"""

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
    """Create or replace the range assigned to a player slot.

    Accepts a range expression and PLO mode. Replacing an occupied
    slot discards its previous range and filter configurations.

    If a community board is already selected, hands conflicting
    with that board are removed from the new range.

    Returns:
        HTTP 200 with the number of hands in the resulting range.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the player index or range is invalid.
    """

    with editable_session(session):
        try:
            session.manager.add_range(
                request.hands,
                request.mode,
                player_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Report the range after applying the existing board, if any,
        # rather than the original number of encoded hands.
        added_hands = session.manager.players[player_id].current_range.hands

        return RangeResponse(hands=len(added_hands))


@router.delete("/{player_id}", status_code=204)
def delete_range(player_id: PlayerId, session: SessionDep):
    """Remove the range assigned to a player slot.

    Clears the player's range and all associated street and filter
    states without affecting other player slots or the shared board.

    Removing an already empty slot is allowed.

    Returns:
        HTTP 204 with no response body.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the player index is outside 0-5.
    """

    with editable_session(session):
        try:
            session.manager.remove_range(player_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
