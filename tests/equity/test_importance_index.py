"""Indexed compatible-hand selector: bit boundaries, conditional weights, and scan fallback."""

import random

import numpy as np
import pytest
from numba.typed import List

from plorangelab.encoder import build_mask, encode_board
from plorangelab.equity import compute_equity, importance_sampling
from plorangelab.equity import engine as api_module
from plorangelab.equity.importance_sampling import (
    _build_compatibility_index,
    _index_one_range,
    _run_importance_batch,
    _select_indexed_hand,
)

from ._helpers import _weighted_players

# Deterministic index test data

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


# Lookup table boundary cases

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

def test_indexed_selector_handles_no_compatible_hands() -> None:
    masks = _candidate_masks(9)
    probabilities = np.ones(9, dtype=np.float64) / 9
    occupied = np.uint64((1 << 52) - 1)

    for seed in (0, 7, 29):
        assert _indexed_selector(
            masks, probabilities, occupied, seed=seed
        ) == (-1, 0.0)


# Index parity and fallback

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

def test_scan_fallback_matches_indexed_public_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "_probe_collision_acceptance",
        lambda *args: 0,
    )

    kwargs = {
        "players": _weighted_players(4),
        "mode": 4,
        "board": encode_board("2c3d5h7s9c"),
        "method": "monte-carlo",
        "min_samples": 2_000,
        "max_samples": 2_000,
        "batch_size": 500,
        "target_moe": 0.0,
        "seed": 73,
    }

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

    for indexed_equity, fallback_equity in zip(
        indexed_result.equities,
        fallback_result.equities,
        strict=True,
    ):
        assert indexed_equity == pytest.approx(
            fallback_equity,
            abs=1e-12,
        )
