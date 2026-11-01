"""Adaptive rejection Monte Carlo: precision stopping, cancellation, and resource budgets."""

from threading import Event

import numpy as np
import pytest

import plorangelab.equity.engine as equity_module
import plorangelab.equity.monte_carlo as mc_module
from plorangelab.equity import compute_equity
from plorangelab.rank_combination import RankLookup

from ._helpers import _configured_ranges, _guaranteed_split_players

# Precision stopping and unweighted uncertainty

def test_adaptive_monte_carlo_stops_when_target_is_reached(rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()

    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        target_moe=0.001,
        min_samples=4,
        max_samples=12,
        batch_size=4,
        seed=13,
    rank_lookup=rank_lookup)

    assert result.method == "monte-carlo"
    assert result.stop_reason == "target_reached"
    assert result.simulations == 4
    assert result.attempts == 4
    assert result.equities == [0.5, 0.5]
    assert result.margins_of_error == [0.0, 0.0]

def test_adaptive_stops_at_max_samples_when_precision_is_not_reached(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    sample_index = 0

    def varying_batch(**kwargs: object):
        nonlocal sample_index
        count = int(kwargs["batch_size"])
        first_player_shares = np.asarray(
            [1.0 if (sample_index + offset) % 2 == 0 else 0.0 for offset in range(count)]
        )
        sample_index += count
        first_sum = float(first_player_shares.sum())
        second_sum = float(count - first_sum)
        return (
            np.asarray([first_sum, second_sum]),
            np.asarray(
                [float(np.square(first_player_shares).sum()), second_sum]
            ),
            count,
            count,
        )

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", varying_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        target_moe=0.0,
        min_samples=2,
        max_samples=5,
        batch_size=2,
        seed=13,
    rank_lookup=rank_lookup)

    assert result.stop_reason == "max_simulations"
    assert result.simulations == 5
    assert result.attempts == 5
    assert result.margins_of_error[0] > 0
    assert result.margins_of_error[1] > 0

def test_equity_statistics_include_split_pots_and_sample_variance() -> None:
    # Observations are (1, 0), (0.5, 0.5), and (0, 1): one split pot.
    equities, margins = mc_module._equity_statistics(
        np.asarray([1.5, 1.5]),
        np.asarray([1.25, 1.25]),
        simulations=3,
        confidence_multiplier=1.96,
    )

    expected_moe = 1.96 * np.sqrt(0.25 / 3)
    assert equities == [0.5, 0.5]
    assert margins[0] == pytest.approx(expected_moe)
    assert margins[1] == pytest.approx(expected_moe)


# Cancellation and partial estimates

def test_adaptive_cancellation_before_execution_raises(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    cancelled.set()

    def unexpected_batch(**kwargs: object):
        raise AssertionError("the batch kernel must not run after pre-cancellation")

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", unexpected_batch)
    with pytest.raises(InterruptedError, match="before any simulations"):
        equity_module.compute_equity(
            players,
            mode=4,
            board=board,
            method="monte-carlo",
            min_samples=2,
            max_samples=4,
            batch_size=2,
            cancel_event=cancelled,
        rank_lookup=rank_lookup)

def test_adaptive_cancellation_between_batches_returns_partial_estimate(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    calls = 0

    def tied_batch(**kwargs: object):
        nonlocal calls
        calls += 1
        count = int(kwargs["batch_size"])
        cancelled.set()
        return (
            np.asarray([count * 0.5, count * 0.5]),
            np.asarray([count * 0.25, count * 0.25]),
            count,
            count,
        )

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", tied_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=4,
        max_samples=8,
        batch_size=2,
        cancel_event=cancelled,
    rank_lookup=rank_lookup)

    assert calls == 1
    assert result.stop_reason == "cancelled"
    assert result.simulations == 2
    assert result.equities == [0.5, 0.5]


# Attempt and memory budgets

def test_attempt_limit_returns_partial_adaptive_batch_without_losing_successes(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    calls = 0

    def exhausting_batch(**kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                np.asarray([1.0, 0.0]),
                np.asarray([1.0, 0.0]),
                1,
                9_999,
            )
        assert kwargs["attempt_budget"] == 1
        return (
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            0,
            1,
        )

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", exhausting_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=1,
        max_samples=4,
        batch_size=2,
    rank_lookup=rank_lookup)

    assert result.stop_reason == "attempt_limit"
    assert result.simulations == 1
    assert result.attempts == 10_000
    assert result.equities == [1.0, 0.0]

def test_fixed_budget_memory_is_bounded_by_batch_size(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _configured_ranges(2, mode=4, multi_hand=False)
    original_batch = mc_module._run_monte_carlo_batch
    allocated_batch_sizes: list[int] = []

    def recording_batch(**kwargs: object):
        allocated_batch_sizes.append(int(kwargs["batch_size"]))
        return original_batch(**kwargs)

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", recording_batch)
    result = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=11,
        max_samples=11,
        target_moe=0.0,
        seed=17,
        method="monte-carlo",
        batch_size=4,
    rank_lookup=rank_lookup)

    assert sum(allocated_batch_sizes) == 11
    assert allocated_batch_sizes == [4, 4, 3]
    assert max(allocated_batch_sizes) <= 4
    assert sum(result.equities) == pytest.approx(1.0)

def test_attempts_are_capped_per_batch_and_global_attempts_accumulate(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    calls = 0

    def capped_batch(**kwargs: object):
        nonlocal calls
        calls += 1
        assert kwargs["attempt_budget"] == mc_module._MAX_BATCH_ATTEMPTS
        if calls == 1:
            return (
                np.asarray([0.0, 0.0]),
                np.asarray([0.0, 0.0]),
                0,
                mc_module._MAX_BATCH_ATTEMPTS,
            )
        cancelled.set()
        return (
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            1,
            1,
        )

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", capped_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        target_moe=0.0,
        min_samples=1,
        max_samples=20_000,
        batch_size=10_000,
        cancel_event=cancelled,
    rank_lookup=rank_lookup)

    assert calls == 2
    assert result.stop_reason == "cancelled"
    assert result.simulations == 1
    assert result.attempts == mc_module._MAX_BATCH_ATTEMPTS + 1
    assert result.equities == [1.0, 0.0]
