from contextlib import asynccontextmanager
from time import monotonic

from fastapi import FastAPI, HTTPException

from plorangelab.bootstrap import load_resources
from plorangelab.equity import SamplingExhaustedError
from plorangelab.services.range_manager import RangeManager
from plorangelab.services.session_manager import Session, SessionManager

from .schemas import (
    BoardRequest,
    EquityDetails,
    EquityRequest,
    EquityResponse,
    RangeRequest,
)

SESSION_TTL = 30 * 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources = load_resources()
    app.state.session_manager = SessionManager(resources)
    yield
    app.state.session_manager.clear()

app = FastAPI(lifespan=lifespan)
  

def get_manager(session_id: str) -> RangeManager:
    session: Session | None = app.state.session_manager.sessions.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="non existent session id")

    now = monotonic()

    if now - session.last_activity > SESSION_TTL:
        del app.state.session_manager.sessions[session_id]
        raise HTTPException(status_code=404, detail="session expired")

    session.last_activity = now
    return session.manager

@app.post("/api/sessions", status_code=201)
def create_session():
    session_id = app.state.session_manager.create_session()
    return {"session_id": session_id}



@app.post("/api/sessions/{session_id}/range")
def add_range(session_id: str, request: RangeRequest):
    manager = get_manager(session_id)

    try:
        manager.add_range(request.hands, request.mode, request.player_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    added_hands = manager.players[request.player_id].active_street.ranges[-1].hands

    return {
        "hands": len(added_hands)
    }

@app.put("/api/sessions/{session_id}/board")
def change_board(session_id: str, request: BoardRequest):
    manager = get_manager(session_id)

    try:
        manager.change_board(request.board)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.delete("/api/sessions/{session_id}/range", status_code=204)
def delete_range(session_id: str, player_id: int):
    manager = get_manager(session_id)

    try:
        manager.remove_range(player_id)    
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/sessions/{session_id}/equity", response_model=EquityResponse)
def calculate_equity(session_id: str, request: EquityRequest):
    manager = get_manager(session_id)

    active_players = [player for player in manager.players if player is not None]

    if len(active_players) < 2:
        raise HTTPException(status_code=422, detail="Minimum 2 players required for the equity calculation")
    
    if any(player.mode != request.mode for player in active_players):
        raise HTTPException(status_code=422, detail=f"Equity mode PLO{request.mode} does not match the player ranges' mode")

    try:
        ids, equity_result = manager.calculate_equity(request.mode, target_moe=request.target_moe)
    except SamplingExhaustedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
        

    equities = {player_id: equity for player_id, equity in zip(ids, equity_result.equities, strict=True)}

    details = EquityDetails(
        method=equity_result.method,
        margins_of_error=equity_result.margins_of_error,
        simulations=equity_result.simulations,
        attempts=equity_result.attempts,
        stop_reason=equity_result.stop_reason,
        sampler=equity_result.sampler,
        effective_samples=equity_result.effective_samples,
    )

    return EquityResponse(equities=equities, details=details)

