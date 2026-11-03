
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event, Lock

from plorangelab.equity import EquityResult, MonteCarloResult
from plorangelab.types import Mode

from .range_manager import RangeManager


@dataclass
class EquityJob:
    cancel_event: Event = field(default_factory=Event)
    future: Future[tuple[list[int], EquityResult]] | None = None
    progress: MonteCarloResult | None = None
    lock: Lock = field(default_factory=Lock)

    def update_progress(self, update: MonteCarloResult) -> None:
        with self.lock:
            self.progress = update

    def get_progress(self) -> MonteCarloResult | None:
        with self.lock:
            return self.progress

    def cancel(self) -> None:
        self.cancel_event.set()


def start_equity_job(
    manager: RangeManager,
    executor: ThreadPoolExecutor,
    mode: Mode,
    target_moe: float,
    notify: Callable[[MonteCarloResult], None],
    on_finished: Callable[[], None] | None = None,
) -> EquityJob:
    job = EquityJob()

    def on_progress(update: MonteCarloResult) -> None:
        job.update_progress(update)
        notify(update)

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
    return job

