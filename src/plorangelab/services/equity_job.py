"""Run equity calculations as asynchronous thread-pool jobs.

Tracks execution status, progress updates, cooperative cancellation,
and a single subscriber notified when progress or job status changes.
"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Event, Lock
from typing import Literal

from plorangelab.equity import EquityResult, MonteCarloResult
from plorangelab.types import Mode

from .range_manager import RangeManager

JobStatus = Literal[
    "pending",
    "running",
    "completed",
    "cancelled",
    "failed",
]


Subscriber = Callable[[], None]


@dataclass
class EquityJob:
    """Track the state and notifications of one equity calculation.

    The job stores its execution future, cancellation event, latest
    Monte Carlo progress, and an optional subscriber callback.

    Progress and subscriber access are protected by a lock.
    Cancellation is cooperative: setting the event requests that
    the running equity calculation stop, but does not terminate
    its worker thread.

    Only one subscriber is retained at a time.
    """

    cancel_event: Event = field(default_factory=Event)
    future: Future[tuple[list[int], EquityResult]] | None = None
    progress: MonteCarloResult | None = None
    lock: Lock = field(default_factory=Lock)
    job_id: str = field(default_factory=lambda: token_urlsafe(12))
    _subscriber: Subscriber | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def get_status(self) -> JobStatus:
        """Derive the current job status from its execution future.

        Returns:
            "pending": No future is assigned or execution has not started.
            "running": The worker is executing the calculation.
            "completed": Execution finished successfully.
            "cancelled": The future was cancelled or the calculation
                returned a result with stop_reason="cancelled".
            "failed": Execution finished with an exception.

        Notes:
            Requesting cancellation does not immediately change the status.
            A running job remains running until its future reaches a
            terminal state.
        """

        future = self.future

        if future is None:
            return "pending"

        if future.cancelled():
            return "cancelled"

        if future.running():
            return "running"

        if not future.done():
            return "pending"

        if future.exception() is not None:
            return "failed"

        _, result = future.result()

        if result.stop_reason == "cancelled":
            return "cancelled"

        return "completed"

    def cancel(self) -> None:
        """Request cooperative cancellation of the equity calculation.

        Signals the cancellation event checked by the equity engine.
        Does not call Future.cancel() or wait for the worker to finish.
        """

        self.cancel_event.set()

    def subscribe(self, callback: Subscriber) -> None:
        """Register a subscriber for progress and completion notifications.

        Replaces any existing subscriber. If a different subscriber was
        previously registered, invokes its callback to wake it so it
        can detect that its subscription has been replaced.

        The callback is invoked outside the job's lock.

        Args:
            callback: No-argument notification callback.

        Notes:
            Registering a subscriber does not immediately send the
            current progress or status.
        """

        with self.lock:
            previous = self._subscriber
            self._subscriber = callback

        # Wake the displaced subscriber outside the lock so it can
        # detect replacement without risking reentrant lock acquisition.
        if previous is not None and previous is not callback:
            previous()

    def unsubscribe(self, callback: Subscriber) -> None:
        with self.lock:
            if self._subscriber is callback:
                self._subscriber = None

    def is_subscriber(self, callback: Subscriber) -> bool:
        with self.lock:
            return self._subscriber is callback

    def notify_subscriber(self) -> None:
        """Notify the currently registered subscriber, if any.

        Copies the subscriber reference while holding the lock, then
        invokes the callback outside the lock to avoid holding it
        during arbitrary callback execution.
        """

        with self.lock:
            callback = self._subscriber

        if callback is not None:
            callback()

    def update_progress(self, update: MonteCarloResult) -> None:
        """Store the latest Monte Carlo progress and notify the subscriber.

        The progress snapshot is replaced under the job lock.
        Subscriber notification occurs after releasing the lock.
        """

        with self.lock:
            self.progress = update
        self.notify_subscriber()

    def get_progress(self) -> MonteCarloResult | None:
        with self.lock:
            return self.progress


def start_equity_job(
    manager: RangeManager,
    executor: ThreadPoolExecutor,
    mode: Mode,
    target_moe: float,
    on_finished: Callable[[], None] | None = None,
) -> EquityJob:
    """Submit an equity calculation to an existing thread pool.

    Creates an EquityJob and submits a worker that calls the
    provided RangeManager's calculate_equity() method.

    Progress reports are stored on the job and forwarded to its
    subscriber. The job's cancellation event is passed to the
    equity engine.

    The optional on_finished callback runs when the worker exits,
    including when the calculation raises an exception.
    A future completion callback notifies the subscriber after
    the job reaches a terminal state.

    Args:
        manager: RangeManager used to perform the calculation.
        executor: Existing thread pool responsible for execution.
        mode: PLO variant used for the calculation.
        target_moe: Requested Monte Carlo margin of error.
        on_finished: Optional callback invoked when the worker exits.

    Returns:
        EquityJob containing the submitted calculation's future.

    Notes:
        This function does not own or shut down the executor.
        The supplied RangeManager is passed to the worker by reference.
    """

    job = EquityJob()

    def on_progress(update: MonteCarloResult) -> None:
        job.update_progress(update)

    def run():
        try:
            return manager.calculate_equity(
                mode=mode,
                target_moe=target_moe,
                on_progress=on_progress,
                cancel_event=job.cancel_event,
            )
        finally:
            if on_finished is not None:
                on_finished()

    job.future = executor.submit(run)
    # Signal completion independently of progress updates, including
    # calculations that finish without reporting any progress.
    job.future.add_done_callback(lambda _: job.notify_subscriber())
    return job
