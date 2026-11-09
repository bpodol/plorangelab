import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from plorangelab.bootstrap import load_resources
from plorangelab.services.session_manager import Session, SessionManager

from .equity_ws import router as equity_wc_router
from .schemas import (
    BoardRequest,
    BoardResponse,
    RangeRequest,
    RangeResponse,
    SessionResponse,
)

SESSION_TTL = 30 * 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources = load_resources()
    app.state.session_manager = SessionManager(resources)
    app.state.executor = ThreadPoolExecutor(max_workers=2)
    try:
        yield
    finally:
        app.state.session_manager.clear()

        await asyncio.to_thread(
            app.state.executor.shutdown,
            wait=True,
            cancel_futures=True,
        )

        

app = FastAPI(lifespan=lifespan)
app.include_router(equity_wc_router)
  

def get_session(session_id: str) -> Session:
    session = app.state.session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return session


@app.post("/api/sessions", status_code=201, response_model=SessionResponse)
def create_session():
    session_id = app.state.session_manager.create_session()
    return SessionResponse(session_id=session_id)



@app.post("/api/sessions/{session_id}/range", response_model=RangeResponse)
def add_range(session_id: str, request: RangeRequest):
    session = get_session(session_id)

    with session.lock:
        if session.closed:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        if session.is_job_running():
            raise HTTPException(status_code=409, detail="Cannot modify ranges during equity calculation")
        
        try:
            session.manager.add_range(request.hands, request.mode, request.player_id)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        added_hands = session.manager.players[request.player_id].current_range.hands

        return RangeResponse(hands=len(added_hands))

@app.put("/api/sessions/{session_id}/board", response_model=BoardResponse)
def change_board(session_id: str, request: BoardRequest):
    session = get_session(session_id)

    with session.lock:
        if session.closed:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        if session.is_job_running():
            raise HTTPException(status_code=409, detail="Cannot modify board during equity calculation") 
        try:
            session.manager.change_board(request.board)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        players = [
            len(player.current_range.hands) 
            if player is not None else 0
            for player in session.manager.players
        ]

        return BoardResponse(players=players)

@app.delete("/api/sessions/{session_id}/range", status_code=204)
def delete_range(session_id: str, player_id: int):
    session = get_session(session_id)
    with session.lock:
        if session.closed:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        if session.is_job_running():
            raise HTTPException(status_code=409, detail="Cannot modify ranges during equity calculation")
        
        try:
            session.manager.remove_range(player_id)    
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))



