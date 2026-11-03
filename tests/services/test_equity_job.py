"""Tests for equity-job execution, progress reporting, and cancellation."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from plorangelab.services.equity_job import EquityJob, start_equity_job


def test_jobs_have_independent_state() -> None:
    first = EquityJob()
    second = EquityJob()
    update = object()

    first.update_progress(update)
    first.cancel()

    assert first.get_progress() is update
    assert first.cancel_event.is_set()
    assert second.get_progress() is None
    assert not second.cancel_event.is_set()


def test_start_job_forwards_progress_and_returns_result() -> None:
    manager = Mock()
    notify = Mock()
    on_finished = Mock()
    update = object()
    result = object()

    def calculate_equity(*, mode, target_moe, on_progress, cancel_event):
        assert mode == 4
        assert target_moe == 0.002
        assert not cancel_event.is_set()
        on_progress(update)
        return [0, 4], result

    manager.calculate_equity.side_effect = calculate_equity

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = start_equity_job(
            manager=manager,
            executor=executor,
            mode=4,
            target_moe=0.002,
            notify=notify,
            on_finished=on_finished,
        )
        assert job.future is not None
        assert job.future.result(timeout=3) == ([0, 4], result)

    assert job.get_progress() is update
    notify.assert_called_once_with(update)
    on_finished.assert_called_once_with()
    manager.calculate_equity.assert_called_once()


def test_cancel_signals_running_calculation() -> None:
    manager = Mock()
    started = Event()

    def calculate_equity(*, cancel_event, **_kwargs):
        started.set()
        assert cancel_event.wait(timeout=3), "The worker did not receive cancellation"
        return [0, 1], object()

    manager.calculate_equity.side_effect = calculate_equity

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = start_equity_job(
            manager=manager,
            executor=executor,
            mode=4,
            target_moe=0.002,
            notify=Mock(),
        )
        assert job.future is not None
        assert started.wait(timeout=3), "The worker did not start"
        assert not job.cancel_event.is_set()

        job.cancel()

        assert job.cancel_event.is_set()
        assert job.future.result(timeout=3)[0] == [0, 1]


def test_on_finished_runs_if_calculation_fails() -> None:
    manager = Mock()
    manager.calculate_equity.side_effect = RuntimeError("engine failed")
    on_finished = Mock()

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = start_equity_job(
            manager=manager,
            executor=executor,
            mode=4,
            target_moe=0.002,
            notify=Mock(),
            on_finished=on_finished,
        )
        assert job.future is not None
        with pytest.raises(RuntimeError, match="engine failed"):
            job.future.result(timeout=3)

    on_finished.assert_called_once_with()
