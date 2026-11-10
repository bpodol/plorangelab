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
        self.cancel_event.set()

    def subscribe(self, callback: Subscriber) -> None:
        with self.lock:
            previous = self._subscriber
            self._subscriber = callback

        # Wake the previous subscriber so it can exit.
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
        with self.lock:
            callback = self._subscriber

        if callback is not None:
            callback()

    def update_progress(self, update: MonteCarloResult) -> None:
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
    job.future.add_done_callback(lambda _: job.notify_subscriber())
    return job
