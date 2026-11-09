
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from plorangelab.bootstrap import load_resources
from plorangelab.services.session_manager import SessionManager

from .routers.equity_ws import router as equity_ws_router
from .routers.players import router as players_router
from .routers.sessions import router as sessions_router


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

app.include_router(sessions_router)
app.include_router(players_router)
app.include_router(equity_ws_router)




