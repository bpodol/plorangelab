"""Importance sampling: statistical accuracy, multi-street behavior, and dead ends."""

import numpy as np
import pytest
from numba.typed import List

from plorangelab.encoder import build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity
from plorangelab.equity import engine as api_module
from plorangelab.equity.importance_sampling import (
    _build_compatibility_index,
    _importance_statistics,
    _run_importance_batch,
)

from ._helpers import RIVER, _weighted_players

# Exact/rejection agreement and statistical properties

@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize(
    "board_text",
    ["2c3d5h7s9c", "2c3d5h7s"],
)
def test_indexed_importance_matches_exact_over_multiple_seeds(
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    board_text: str,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    players = _weighted_players(mode)
    board = encode_board(board_text)

    exact = compute_equity(
        players, mode=mode, board=board, method="exact"
    )

    estimates = []

    for seed in (11, 29, 47):
        result = compute_equity(
            players,
            mode=mode,
            board=board,
            method="monte-carlo",
            min_samples=4_000,
            max_samples=4_000,
            batch_size=1_000,
            target_moe=0.0,
            seed=seed,
        )

        assert result.sampler == "importance"
        assert result.simulations == 4_000
        assert result.effective_samples is not None

        assert (
            0 < result.effective_samples
            <= result.simulations + 1e-7
        )

        assert sum(result.equities.values()) == pytest.approx(
            1.0, abs=1e-10
        )

        estimates.append(list(result.equities.values()))

    mean_equity = np.mean(estimates, axis=0)

    np.testing.assert_allclose(
        mean_equity,
        list(exact.equities.values()),
        atol=0.025,
        rtol=0,
    )

def test_indexed_and_rejection_samplers_agree_statistically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players = _weighted_players(4)
    board = encode_board("2c3d5h7s9c")

    common = {
        "players": players,
        "mode": 4,
        "board": board,
        "method": "monte-carlo",
        "min_samples": 10_000,
        "max_samples": 10_000,
        "target_moe": 0.0,
        "seed": 101,
    }

    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )
    importance = compute_equity(**common)

    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 1024,
    )
    rejection = compute_equity(**common)

    assert importance.sampler == "importance"
    assert rejection.sampler == "rejection"

    for label in importance.equities:
        assert importance.equities[label] == pytest.approx(
            rejection.equities[label],
            abs=0.035,
        )


# PLO modes, streets, and incompatible selections

@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize(
    "board_text",
    [None, "2c3d5h", "2c3d5h7s", "2c3d5h7s9c"],
)
def test_indexed_importance_supports_modes_and_streets(
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    board_text: str | None,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    board = None if board_text is None else encode_board(board_text)

    result = compute_equity(
        _weighted_players(mode),
        mode=mode,
        board=board,
        method="monte-carlo",
        min_samples=1_000,
        max_samples=1_000,
        batch_size=250,
        seed=5,
        target_moe=0.0,
    )

    assert result.sampler == "importance"
    assert result.simulations == 1_000
    assert result.attempts >= result.simulations
    assert result.effective_samples is not None

    assert (
        0 < result.effective_samples
        <= result.simulations + 1e-7
    )

    assert all(
        np.isfinite(value) and 0 <= value <= 1
        for value in result.equities.values()
    )

    assert sum(result.equities.values()) == pytest.approx(
        1.0, abs=1e-10
    )

def test_importance_batch_handles_dead_end_without_invalid_selection() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("AsQsJcTc", 4),  # Always collides with player 1.
        encode_range("6c6d8h8s", 4),
    ]

    board = encode_board("2c3d5h7s9c")
    hands = List([p.hands for p in players])
    masks = List([p.masks for p in players])

    probabilities = List([
        p.weights.astype(np.float64) / p.weights.sum()
        for p in players
    ])
    cumulative = List([np.cumsum(p) for p in probabilities])

    membership, lookup, enabled = _build_compatibility_index(
        masks, probabilities
    )
    assert enabled

    result = _run_importance_batch(
        hands,
        masks,
        probabilities,
        cumulative,
        4,
        board,
        build_mask(board),
        10,
        25,
        np.random.default_rng(12),
        membership,
        lookup,
        True,
    )

    assert result[-2:] == (0, 25)
    assert result[0] == 0.0
    assert result[1] == 0.0

    for totals in result[2:5]:
        np.testing.assert_array_equal(totals, np.zeros(3))


# Importance-weighted equity and uncertainty

def test_importance_sampler_agrees_with_exact_weighted_colliding_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plorangelab.equity import engine as api_module

    board = encode_board(RIVER)

    players = [
        encode_range("AsAhKdQc@70,QsQhJdTd@20,6h6d4s4h@10", 4),
        encode_range("AsQsJcTc@50,KsKhJcTc@50", 4),
        encode_range("KsQc8c8d@55,JsJh7d7h@45", 4),
    ]

    exact = compute_equity(
        players,
        mode=4,
        board=board,
        method="exact",
    )

    # Force the importance-sampling path.
    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=20_000,
        max_samples=20_000,
        batch_size=2_000,
        seed=103,
        target_moe=0.0,
    )

    assert sampled.sampler == "importance"
    assert sampled.simulations == 20_000
    assert 1 < sampled.effective_samples <= sampled.simulations

    assert sum(sampled.equities.values()) == pytest.approx(1.0)

    for name in exact.equities:
        assert sampled.equities[name] == pytest.approx(
            exact.equities[name],
            abs=0.02,
        )

def test_importance_statistics_account_for_unequal_sample_weights() -> None:

    equities, margins, ess = _importance_statistics(
        4.0,
        10.0,
        np.array([1.0, 3.0]),
        np.array([1.0, 9.0]),
        np.array([1.0, 9.0]),
        1.96,
    )

    assert equities == {
        "player 1:": pytest.approx(0.25),
        "player 2:": pytest.approx(0.75),
    }

    assert ess == pytest.approx(1.6)

    assert margins == {
        "player 1:": None,
        "player 2:": None,
    }
