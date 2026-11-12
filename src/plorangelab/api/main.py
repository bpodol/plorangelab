"""Initialize the FastAPI application and manage shared resources.

Loads equity engine resources, creates the session registry and
thread pool, registers API routers, and coordinates shutdown.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from plorangelab.bootstrap import load_resources
from plorangelab.services.session_manager import SessionManager

from .routers.equity_ws import router as equity_ws_router
from .routers.filters import router as filters_router
from .routers.players import router as players_router
from .routers.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources and release them on application shutdown.

    Startup:
        Load precomputed equity engine resources.
        Create the in-memory SessionManager.
        Create a ThreadPoolExecutor with two workers.

    Shutdown:
        Close all registered sessions and request cancellation
        of their unfinished equity calculations.
        Shut down the executor, cancelling queued futures and
        waiting for running workers to finish.

    Resources are stored on app.state for use by API handlers.

    Notes:
        Running equity calculations use cooperative cancellation.
        Executor shutdown waits for running workers to exit rather
        than forcibly terminating their threads.
    """

    resources = load_resources()

    app.state.session_manager = SessionManager(resources)
    app.state.executor = ThreadPoolExecutor(max_workers=2)

    try:
        yield
    finally:
        # Request cancellation through session cleanup before shutting
        # down the executor and waiting for running workers to finish.
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
app.include_router(filters_router)
