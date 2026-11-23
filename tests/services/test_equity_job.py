"""Tests for equity-job execution, progress reporting, and cancellation."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from plorangelab.equity import EquityResult
from plorangelab.equity.results import StopReason
from plorangelab.services.equity_job import EquityJob, start_equity_job


def _equity_result(*, stop_reason: StopReason | None = None) -> EquityResult:
    """Create a representative player-indexed Monte Carlo result."""
    return EquityResult(
        equities={0: 0.6, 4: 0.4},
        method="monte-carlo",
        margins_of_error={0: 0.01, 4: 0.01},
        simulations=100,
        attempts=120,
        stop_reason=stop_reason,
        sampler="rejection",
    )


def test_jobs_have_independent_state() -> None:
    first = EquityJob()
    second = EquityJob()
    update = _equity_result()

    first.update_progress(update)
    first.cancel()

    assert first.get_progress() is update
    assert first.cancel_event.is_set()
    assert second.get_progress() is None
    assert not second.cancel_event.is_set()


def test_get_snapshot_tracks_status_and_latest_progress() -> None:
    job = EquityJob()
    initial = job.get_snapshot()
    assert initial.status == "pending"
    assert initial.progress is None

    future: Future[EquityResult] = Future()
    job.future = future
    assert future.set_running_or_notify_cancel()

    first_progress = _equity_result()
    job.update_progress(first_progress)
    first_snapshot = job.get_snapshot()
    assert first_snapshot.status == "running"
    assert first_snapshot.progress is first_progress

    latest_progress = _equity_result()
    job.update_progress(latest_progress)
    latest_snapshot = job.get_snapshot()
    assert latest_snapshot.status == "running"
    assert latest_snapshot.progress is latest_progress
    # Previously returned snapshots do not change when progress is replaced.
    assert first_snapshot.progress is first_progress

    future.set_result(latest_progress)
    completed_snapshot = job.get_snapshot()
    assert completed_snapshot.status == "completed"
    assert completed_snapshot.progress is latest_progress
    assert first_snapshot.status == "running"


def test_start_job_forwards_progress_and_returns_result() -> None:
    manager = Mock()
    on_finished = Mock()
    update = _equity_result()
    result = _equity_result()

    def calculate_equity(*, target_moe, on_progress, cancel_event):
        assert target_moe == 0.002
        assert not cancel_event.is_set()
        on_progress(update)
        return result

    manager.calculate_equity.side_effect = calculate_equity

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = start_equity_job(
            manager=manager,
            executor=executor,
            target_moe=0.002,
            on_finished=on_finished,
        )
        assert job.future is not None
        assert job.future.result(timeout=3) is result

    assert job.get_progress() is update
    assert job.get_status() == "completed"
    on_finished.assert_called_once_with()
    manager.calculate_equity.assert_called_once()


def test_cancel_signals_running_calculation() -> None:
    manager = Mock()
    started = Event()
    expected_result = _equity_result(stop_reason="cancelled")

    def calculate_equity(*, cancel_event, **_kwargs):
        started.set()
        assert cancel_event.wait(timeout=3), "The worker did not receive cancellation"
        return expected_result

    manager.calculate_equity.side_effect = calculate_equity

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = start_equity_job(
            manager=manager,
            executor=executor,
            target_moe=0.002,
        )
        assert job.future is not None
        assert started.wait(timeout=3), "The worker did not start"
        assert not job.cancel_event.is_set()

        job.cancel()

        assert job.cancel_event.is_set()
        assert job.future.result(timeout=3) is expected_result
        assert job.get_status() == "cancelled"


def test_on_finished_runs_if_calculation_fails() -> None:
    manager = Mock()
    manager.calculate_equity.side_effect = RuntimeError("engine failed")
    on_finished = Mock()

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = start_equity_job(
            manager=manager,
            executor=executor,
            target_moe=0.002,
            on_finished=on_finished,
        )
        assert job.future is not None
        with pytest.raises(RuntimeError, match="engine failed"):
            job.future.result(timeout=3)
        assert job.get_status() == "failed"

    on_finished.assert_called_once_with()
