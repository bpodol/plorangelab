
"""Regression tests for the collision-aware Monte Carlo compatibility index."""

import random

import numpy as np
import pytest
from numba.typed import List

from plorangelab.encoder import build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity
from plorangelab.equity import api as api_module
from plorangelab.equity import importance_sampling 
from plorangelab.equity.importance_sampling import (
    _build_compatibility_index, 
    _index_one_range, _select_indexed_hand, 
    _run_importance_batch,
    _INDEX_MAX_HANDS,
)


def _candidate_masks(count: int, mode: int = 5) -> np.ndarray:
    """Generate distinct, reproducible hands spanning different card masks."""
    rng = random.Random(2026 + mode)
    seen: set[tuple[int, ...]] = set()

    while len(seen) < count:
        seen.add(tuple(sorted(rng.sample(range(52), mode))))

    return np.asarray(
        [
            build_mask(np.asarray(hand, dtype=np.uint8))
            for hand in sorted(seen)
        ],
        dtype=np.uint64,
    )


def _indexed_selector(
    masks: np.ndarray,
    probabilities: np.ndarray,
    occupied: np.uint64,
    *,
    seed: int = 42,
):
    membership, lookup = _index_one_range(masks, probabilities)
    blocked = np.empty(membership.shape[1], dtype=np.uint64)
    legal = np.empty(lookup.shape[0], dtype=np.uint8)

    return _select_indexed_hand(
        occupied,
        probabilities,
        membership,
        lookup,
        blocked,
        legal,
        np.random.default_rng(seed),
    )


def _weighted_players(mode: int):
    if mode == 4:
        texts = (
            "AsAhKdQc@80,QsQhJdTd@20",
            "KsKhJcTc@35,AsQsJcTc@65",
            "AdAcTsTh@60,KdKh8c8d@40",
        )
    else:
        texts = (
            "AsAhKdQc6c@80,QsQhJdTd6d@20",
            "KsKhJcTc8s@35,AsQsJcTc8h@65",
            "AdAcTsTh4s@60,KdKh8c8d4d@40",
        )

    return [encode_range(text, mode) for text in texts]


# 1. Bitset and 8-hand lookup boundaries

@pytest.mark.parametrize("count", [1, 7, 8, 9, 63, 64, 65, 129])
def test_indexed_mass_and_selection_at_bit_boundaries(count: int) -> None:
    masks = _candidate_masks(count)
    weights = np.arange(1, count + 1, dtype=np.float64) ** 1.5
    probabilities = weights / weights.sum()

    occupied_masks = (
        np.uint64(0),
        masks[0],
        build_mask(np.array([0, 13, 25, 38], dtype=np.uint8)),
        np.uint64((1 << 52) - 1),
    )

    for occupied in occupied_masks:
        index, mass = _indexed_selector(
            masks, probabilities, occupied
        )
        legal = (masks & occupied) == 0

        assert mass == pytest.approx(
            probabilities[legal].sum(), abs=1e-13
        )

        if legal.any():
            assert 0 <= index < count
            assert legal[index]
        else:
            assert (index, mass) == (-1, 0.0)


# 2. Weighted selection frequencies

def test_indexed_selector_matches_conditional_weighted_frequencies() -> None:
    masks = _candidate_masks(17)
    probabilities = np.arange(1, 18, dtype=np.float64) ** 2
    probabilities /= probabilities.sum()

    occupied = build_mask(np.array([0, 12], dtype=np.uint8))
    legal = (masks & occupied) == 0
    expected_mass = float(probabilities[legal].sum())

    assert legal.sum() >= 2

    membership, lookup = _index_one_range(
        masks, probabilities
    )
    blocked = np.empty(membership.shape[1], dtype=np.uint64)
    legal_blocks = np.empty(lookup.shape[0], dtype=np.uint8)

    rng = np.random.default_rng(123)
    counts = np.zeros(len(masks), dtype=np.int64)
    trials = 20_000

    for _ in range(trials):
        selected, mass = _select_indexed_hand(
            occupied,
            probabilities,
            membership,
            lookup,
            blocked,
            legal_blocks,
            rng,
        )

        assert mass == pytest.approx(expected_mass, abs=1e-13)
        counts[selected] += 1

    assert not np.any(counts[~legal])

    observed = counts[legal] / trials
    expected = probabilities[legal] / expected_mass

    np.testing.assert_allclose(
        observed, expected, atol=0.015, rtol=0
    )


# 3. Completely blocked ranges

def test_indexed_selector_handles_no_compatible_hands() -> None:
    masks = _candidate_masks(9)
    probabilities = np.ones(9, dtype=np.float64) / 9
    occupied = np.uint64((1 << 52) - 1)

    for seed in (0, 7, 29):
        assert _indexed_selector(
            masks, probabilities, occupied, seed=seed
        ) == (-1, 0.0)


# 4. Indexed kernel vs original scanning kernel

@pytest.mark.parametrize("mode", [4, 5])
def test_indexed_batch_matches_original_scan_with_same_seed(
    mode: int,
) -> None:
    players = _weighted_players(mode)
    board = encode_board("2c3d5h7s9c")

    hands = List([player.hands for player in players])
    masks = List([player.masks for player in players])

    probabilities = List([
        p.weights.astype(np.float64) / p.weights.sum()
        for p in players
    ])

    cumulative = List([
        np.cumsum(p) for p in probabilities
    ])

    membership, lookup, enabled = _build_compatibility_index(
        masks, probabilities
    )
    assert enabled

    args = (
        hands,
        masks,
        probabilities,
        cumulative,
        mode,
        board,
        build_mask(board),
        1_000,
        20_000,
    )

    scan = _run_importance_batch(
        *args,
        np.random.default_rng(81),
        membership,
        lookup,
        False,
    )

    indexed = _run_importance_batch(
        *args,
        np.random.default_rng(81),
        membership,
        lookup,
        True,
    )

    assert indexed[-2:] == scan[-2:]

    for actual, expected in zip(indexed[:-2], scan[:-2]):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-10
        )


# 5. Memory guard and fallback

def test_index_guard_falls_back_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importance_sampling, "_INDEX_MAX_HANDS", 8)

    probabilities = List([
        np.full(8, 1 / 8, dtype=np.float64),
        np.full(9, 1 / 9, dtype=np.float64),
    ])

    masks = List([
        _candidate_masks(8),
        _candidate_masks(9),
    ])

    membership, lookup, enabled = _build_compatibility_index(
        masks, probabilities
    )

    assert not enabled

    for arr in membership:
        assert arr.shape == (0, 0)

    for arr in lookup:
        assert arr.shape == (0, 0)


# 6. Statistical correctness against exact equity

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


# 7. Indexed importance vs rejection sampling

def test_indexed_and_rejection_samplers_agree_statistically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players = _weighted_players(4)
    board = encode_board("2c3d5h7s9c")

    common = dict(
        players=players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=10_000,
        max_samples=10_000,
        target_moe=0.0,
        seed=101,
    )

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


# 8. PLO4/PLO5 and all streets

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


# 9. Dead-end selection does not evaluate invalid hands

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


# 10. Full scan fallback preserves numerical results

def test_scan_fallback_matches_indexed_public_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    kwargs = dict(
        players=_weighted_players(4),
        mode=4,
        board=encode_board("2c3d5h7s9c"),
        method="monte-carlo",
        min_samples=2_000,
        max_samples=2_000,
        batch_size=500,
        target_moe=0.0,
        seed=73,
    )

    indexed_result = compute_equity(**kwargs)

    monkeypatch.setattr(importance_sampling, "_INDEX_MAX_HANDS", 1)
    fallback_result = compute_equity(**kwargs)

    assert indexed_result.sampler == "importance"
    assert fallback_result.sampler == "importance"

    assert (
        indexed_result.simulations
        == fallback_result.simulations
        == 2_000
    )

    for name in indexed_result.equities:
        assert indexed_result.equities[name] == pytest.approx(
            fallback_result.equities[name],
            abs=1e-12,
        )
