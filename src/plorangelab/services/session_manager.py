from dataclasses import dataclass
from secrets import token_urlsafe
from time import monotonic

from plorangelab.bootstrap import EngineResources

from .range_manager import RangeManager


@dataclass
class Session:
    manager: RangeManager
    last_activity: float


class SessionManager:
    def __init__(self, resources: EngineResources):
        self.sessions: dict[str, Session] = {}
        self.resources = resources


    def create_session(self,) -> str:
        session_id = token_urlsafe(32)
        manager = RangeManager(self.resources)
        now = monotonic()
        self.sessions[session_id] = Session(manager=manager, last_activity=now)

        return session_id

    def delete_session(self, session_id: str) -> None:
        del self.sessions[session_id]

    def clear(self):
        self.sessions.clear()


