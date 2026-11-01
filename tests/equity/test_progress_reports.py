"""Test Monte Carlo progress reporting and callback cancellation."""

from threading import Event

import numpy as np
import pytest

import plorangelab.equity.engine as equity_module
import plorangelab.equity.importance_sampling as importance_module
import plorangelab.equity.monte_carlo as mc_module
from plorangelab.encoder import encode_range
from plorangelab.equity.monte_carlo import MonteCarloResult
from plorangelab.rank_combination import RankLookup

from ._helpers import _guaranteed_split_players


def test_rejection_progress_callback(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    updates: list[MonteCarloResult] = []

    def balanced_batch(**kwargs):
        count = int(kwargs["batch_size"])
        assert count == 2

        # One win per player in each two-simulation batch.
        shares = np.array([1.0, 1.0])

        return shares, shares.copy(), count, count

    monkeypatch.setattr(
        mc_module, "_run_monte_carlo_batch", balanced_batch
    )

    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=2,
        max_samples=6,
        batch_size=2,
        target_moe=0.0,
        on_progress=updates.append,
    rank_lookup=rank_lookup)

    assert len(updates) == 2
    assert [u.simulations for u in updates] == [2, 4]
    assert [u.attempts for u in updates] == [2, 4]

    for update in updates:
        assert update.sampler == "rejection"
        assert update.stop_reason is None
        assert update.equities == [0.5, 0.5]

    # Verify the MOE uses cumulative statistics.
    assert updates[0].margins_of_error[0] == pytest.approx(
        1.96 * np.sqrt(0.5 / 2)
    )
    assert updates[1].margins_of_error[0] == pytest.approx(
        1.96 * np.sqrt((1 / 3) / 4)
    )

    assert result.simulations == 6
    assert result.stop_reason == "max_simulations"


def test_importance_progress_callback(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players = [
        encode_range("AsKsQsJs", 4),
        encode_range("AhKhQhJh", 4),
        encode_range("AdKdQdJd", 4),
    ]

    updates: list[MonteCarloResult] = []

    # Force the importance sampler through the public engine.
    monkeypatch.setattr(
        equity_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    # Index construction is unrelated to progress reporting.
    monkeypatch.setattr(
        importance_module,
        "_build_compatibility_index",
        lambda *args: (None, None, False),
    )

    def balanced_importance_batch(*args, **kwargs):
        assert args[7] == 2  # batch_size

        # Two equal-weight simulations:
        # Player 1 wins once, player 2 wins once.
        shares = np.array([1.0, 1.0, 0.0])

        return (
            2.0,            # Sum of weights
            2.0,            # Sum of squared weights
            shares,
            shares.copy(),
            shares.copy(),
            2,              # Simulations
            2,              # Attempts
        )

    monkeypatch.setattr(
        importance_module,
        "_run_importance_batch",
        balanced_importance_batch,
    )

    result = equity_module.compute_equity(
        players,
        mode=4,
        method="monte-carlo",
        min_samples=6,
        max_samples=1_000,
        batch_size=2,
        target_moe=0.5,
        on_progress=updates.append,
    rank_lookup=rank_lookup)

    assert len(updates) == 2
    assert [u.simulations for u in updates] == [2, 4]
    assert [u.attempts for u in updates] == [2, 4]

    for update in updates:
        assert update.sampler == "importance"
        assert update.stop_reason is None
        assert update.effective_samples == pytest.approx(
            update.simulations
        )

        assert update.equities == [0.5, 0.5, 0]

        assert update.margins_of_error[0] > 0
        assert update.margins_of_error[2] == 0.0

    assert result.sampler == "importance"
    assert result.stop_reason == "target_reached"
    assert result.simulations == 6


def test_cancellation_from_callback_rejection(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()

    cancelled = Event()
    updates: list[MonteCarloResult] = []
    batch_calls = 0

    def balanced_batch(**kwargs):
        nonlocal batch_calls
        batch_calls += 1

        return (
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            2,
            2,
        )

    def on_progress(progress: MonteCarloResult) -> None:
        updates.append(progress)
        cancelled.set()

    monkeypatch.setattr(
        mc_module, "_run_monte_carlo_batch", balanced_batch
    )

    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=2,
        max_samples=10,
        batch_size=2,
        target_moe=0.0,
        cancel_event=cancelled,
        on_progress=on_progress,
    rank_lookup=rank_lookup)

    assert cancelled.is_set()
    assert batch_calls == 1
    assert len(updates) == 1

    assert result.stop_reason == "cancelled"
    assert result.simulations == 2
    assert result.attempts == 2
    assert result.equities == updates[0].equities
    assert result.margins_of_error == updates[0].margins_of_error


def test_cancellation_from_callback_importance(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players = [
        encode_range("AsKsQsJs", 4),
        encode_range("AhKhQhJh", 4),
        encode_range("AdKdQdJd", 4),
    ]

    cancelled = Event()
    updates: list[MonteCarloResult] = []
    batch_calls = 0

    # Force importance sampling.
    monkeypatch.setattr(
        equity_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    monkeypatch.setattr(
        importance_module,
        "_build_compatibility_index",
        lambda *args: (None, None, False),
    )

    def mock_batch(*args, **kwargs):
        nonlocal batch_calls
        batch_calls += 1

        shares = np.array([1.0, 1.0, 0.0])

        return (
            2.0,
            2.0,
            shares,
            shares.copy(),
            shares.copy(),
            2,
            2,
        )

    def on_progress(progress: MonteCarloResult) -> None:
        updates.append(progress)
        cancelled.set()

    monkeypatch.setattr(
        importance_module,
        "_run_importance_batch",
        mock_batch,
    )

    result = equity_module.compute_equity(
        players,
        mode=4,
        method="monte-carlo",
        min_samples=6,
        max_samples=1_000,
        batch_size=2,
        target_moe=0.0,
        cancel_event=cancelled,
        on_progress=on_progress,
    rank_lookup=rank_lookup)

    assert cancelled.is_set()
    assert batch_calls == 1
    assert len(updates) == 1

    assert result.stop_reason == "cancelled"
    assert result.sampler == "importance"
    assert result.simulations == 2
    assert result.attempts == 2
    assert result.effective_samples == pytest.approx(2.0)

    assert result.equities == updates[0].equities
    assert result.margins_of_error == updates[0].margins_of_error


def test_no_progress_for_exact_equity(
    monkeypatch: pytest.MonkeyPatch,
rank_lookup: RankLookup) -> None:
    players, board = _guaranteed_split_players()
    updates: list[MonteCarloResult] = []

    def fake_exact_equity(**kwargs):
        return [0.5, 0.5]

    monkeypatch.setattr(
        equity_module, "run_exact_equity", fake_exact_equity
    )

    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="exact",
        on_progress=updates.append,
    rank_lookup=rank_lookup)

    assert updates == []
    assert result.method == "exact"
    assert result.equities == [0.5, 0.5]
    assert result.simulations is None
    assert result.margins_of_error is None