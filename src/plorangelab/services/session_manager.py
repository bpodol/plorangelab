from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Lock
from time import monotonic

from plorangelab.bootstrap import EngineResources

from .equity_job import EquityJob
from .range_manager import RangeManager

SESSION_TTL = 30 * 60


@dataclass
class Session:
    manager: RangeManager
    last_activity: float
    equity_job: EquityJob | None = None
    lock: Lock = field(default_factory=Lock, repr=False)
    closed: bool = False

    def touch(self) -> None:
        with self.lock:
            if not self.closed:
                self.last_activity = monotonic()

    def is_job_running(self) -> bool:
        job = self.equity_job

        return job is not None and (job.future is None or not job.future.done())


class SessionManager:
    def __init__(self, resources: EngineResources):
        self.sessions: dict[str, Session] = {}
        self.resources = resources
        self.lock = Lock()

    def create_session(
        self,
    ) -> str:
        with self.lock:
            session_id = token_urlsafe(32)
            manager = RangeManager(self.resources)
            now = monotonic()
            self.sessions[session_id] = Session(manager=manager, last_activity=now)

            return session_id

    def get_session(self, session_id: str) -> Session | None:
        with self.lock:
            session = self.sessions.get(session_id)

        if session is None:
            return None

        with session.lock:
            if session.closed:
                return None

            now = monotonic()

            if (
                not session.is_job_running()
                and now - session.last_activity > SESSION_TTL
            ):
                session.closed = True
            else:
                session.last_activity = now
                return session

        # Remove the expired session.
        with self.lock:
            if self.sessions.get(session_id) is session:
                del self.sessions[session_id]

        return None

    def delete_session(self, session_id: str) -> None:
        with self.lock:
            session = self.sessions.get(session_id)

        if session is None:
            return

        with session.lock:
            session.closed = True

            if session.is_job_running():
                session.equity_job.cancel()

        with self.lock:
            if self.sessions.get(session_id) is session:
                del self.sessions[session_id]

    def clear(self) -> None:
        with self.lock:
            ids = list(self.sessions)

        for session_id in ids:
            self.delete_session(session_id)
