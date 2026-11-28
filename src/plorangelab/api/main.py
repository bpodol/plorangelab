"""Initialize the FastAPI application and manage shared resources.

Loads equity engine resources, creates the session registry and
thread pool, registers API routers, and coordinates shutdown.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from plorangelab.bootstrap import load_resources
from plorangelab.config import Settings
from plorangelab.db import create_db_engine, create_session_factory
from plorangelab.services.session_manager import SessionManager

from .exception_handlers import handle_validation_error
from .routers.equity_sse import router as equity_sse_router
from .routers.filters import router as filters_router
from .routers.players import router as players_router
from .routers.sessions import router as sessions_router
from .routers.users import router as users_router


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

    settings = Settings()
    resources = load_resources()

    async with AsyncExitStack() as stack:
        engine = create_db_engine(settings.database_url)
        stack.push_async_callback(engine.dispose)

        db_session_factory = create_session_factory(engine)

        executor = ThreadPoolExecutor(max_workers=2)
        stack.push_async_callback(
            asyncio.to_thread,
            executor.shutdown,
            wait=True,
            cancel_futures=True,
        )

        session_manager = SessionManager(resources)
        stack.callback(session_manager.clear)

        app.state.db_session_factory = db_session_factory
        app.state.session_manager = session_manager
        app.state.executor = executor

        yield


app = FastAPI(lifespan=lifespan)

app.add_exception_handler(
    RequestValidationError,
    handle_validation_error,
)

app.include_router(sessions_router)
app.include_router(players_router)
app.include_router(filters_router)
app.include_router(equity_sse_router)
app.include_router(users_router)
