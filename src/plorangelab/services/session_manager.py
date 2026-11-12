"""Manage in-memory application sessions and their lifecycles.

Each session owns a RangeManager and may reference an equity job.
Sessions expire after a period of inactivity unless an equity
calculation is still running.

Session lookup, expiration, and deletion use locks to coordinate
access to shared state.
"""

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
    """Store the mutable state and lifecycle metadata of one session.

    Attributes:
        manager: RangeManager containing the session's player and board state.
        last_activity: Time of the last recorded activity, measured
            using the monotonic clock.
        equity_job: Current equity job, if one is associated with the session.
        lock: Protects session lifecycle state during coordinated access.
        closed: Whether the session has been invalidated.

    Notes:
        The session lock is not automatically acquired when accessing
        or modifying the RangeManager or equity_job attributes.
        Callers must coordinate concurrent operations separately.
    """

    manager: RangeManager
    last_activity: float
    equity_job: EquityJob | None = None
    lock: Lock = field(default_factory=Lock, repr=False)
    closed: bool = False

    def touch(self) -> None:
        """Refresh the last activity timestamp if the session is open.

        Uses the monotonic clock and updates the timestamp while holding
        the session lock. Closed sessions are not refreshed.
        """

        with self.lock:
            if not self.closed:
                self.last_activity = monotonic()

    def is_job_running(self) -> bool:
        """Check whether an equity job is still pending or executing.

        Returns True when a job exists and its future has not completed,
        or when the job has not yet been assigned a future.

        A cancellation request alone does not make a job finished.
        """

        job = self.equity_job

        return job is not None and (job.future is None or not job.future.done())


class SessionManager:
    """Own and manage the application's in-memory sessions.

    Creates isolated RangeManager instances using shared engine
    resources, tracks sessions by randomly generated identifiers,
    and handles inactivity expiration and explicit deletion.

    The manager lock protects access to the session registry.
    Individual session locks coordinate lifecycle operations.

    Expiration is checked when a session is retrieved; this class
    does not run a background cleanup task.
    """

    def __init__(self, resources: EngineResources):
        self.sessions: dict[str, Session] = {}
        self.resources = resources
        self.lock = Lock()

    def create_session(
        self,
    ) -> str:
        """Create and register a new session.

        Initializes a RangeManager with shared engine resources and
        records the current monotonic time as the last activity timestamp.

        Returns:
            Randomly generated session identifier.
        """

        with self.lock:
            session_id = token_urlsafe(32)
            manager = RangeManager(self.resources)
            now = monotonic()
            self.sessions[session_id] = Session(manager=manager, last_activity=now)

            return session_id

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve an open session and refresh its activity timestamp.

        Returns None when the session does not exist, has been closed,
        or has exceeded the inactivity timeout without a running job.

        Active equity jobs prevent inactivity expiration. Successful
        retrieval refreshes the session's last activity timestamp.

        Expired sessions are marked closed and removed from the registry.

        Args:
            session_id: Identifier returned by create_session().

        Returns:
            The active Session, or None when unavailable.

        Notes:
            Expiration is checked on lookup rather than periodically.
            Returning a Session does not prevent another operation from
            subsequently closing or deleting it.
        """

        with self.lock:
            session = self.sessions.get(session_id)

        if session is None:
            return None

        with session.lock:
            if session.closed:
                return None

            now = monotonic()
            # Keep sessions alive while their equity jobs are unfinished,
            # even if the inactivity timeout has elapsed.
            if (
                not session.is_job_running()
                and now - session.last_activity > SESSION_TTL
            ):
                session.closed = True
            else:
                session.last_activity = now
                return session

        # Remove only the session instance we expired. Avoid deleting
        # a different entry if the registry changed meanwhile.
        with self.lock:
            if self.sessions.get(session_id) is session:
                del self.sessions[session_id]

        return None

    def delete_session(self, session_id: str) -> None:
        """Close and remove a session from the registry.

        Does nothing when the identifier is unknown.

        Marks the session closed and requests cooperative cancellation
        if its equity job has not finished.

        Removes the registry entry only if it still references the
        same Session instance.

        Args:
            session_id: Identifier of the session to delete.

        Notes:
            Deletion does not wait for the equity worker to finish.
            Cancellation is handled cooperatively by the equity engine.
        """

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
        """Delete sessions currently registered with the manager.

        Takes a snapshot of session identifiers, then deletes each
        session through delete_session(), including requesting
        cancellation of unfinished equity jobs.

        Does not wait for running workers to terminate.
        """

        with self.lock:
            ids = list(self.sessions)

        for session_id in ids:
            self.delete_session(session_id)
