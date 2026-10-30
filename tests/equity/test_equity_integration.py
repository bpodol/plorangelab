"""Cross-engine public API contracts: exact vs Monte Carlo, blockers, and immutability."""

import numpy as np
import pytest

from plorangelab.rank_combination import RankLookup

from plorangelab.encoder import encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, TURN, _configured_ranges

# Board blocking and equity conservation

def test_board_blocked_hands_are_removed_before_calculation(rank_lookup: RankLookup) -> None:
    players = [
        encode_range("2cAsAhKd,QsQhJdTd", 4),
        encode_range("KsKhJcTc", 4),
    ]

    equity = compute_equity(
        players, mode=4, board=encode_board(RIVER), method="exact",
        rank_lookup=rank_lookup,
    ).equities

    assert equity == [0.0, 1.0]

@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_equities_conserve_the_pot(method: str, rank_lookup: RankLookup) -> None:
    players, board = _configured_ranges(3, mode=5, multi_hand=True)

    equity = compute_equity(
        players,
        mode=5,
        board=board,
        target_moe=0.0,
        min_samples=50,
        max_samples=50,
        seed=23,
        method=method,  # type: ignore[arg-type]
        rank_lookup=rank_lookup,
    ).equities

    assert sum(equity) == pytest.approx(1.0)


# Exact vs sampled equity

def test_exact_and_monte_carlo_equity_agree_within_sampling_tolerance(rank_lookup: RankLookup) -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board(TURN)

    exact = compute_equity(players, mode=4, board=board, method="exact", rank_lookup=rank_lookup).equities
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=5_000,
        max_samples=5_000,
        target_moe=0.0,
        seed=31,
        method="monte-carlo",
    rank_lookup=rank_lookup).equities

    for sampled_equity, exact_equity in zip(
        sampled, exact, strict=True
    ):
        assert sampled_equity == pytest.approx(
            exact_equity, abs=0.01
        )

def test_exact_and_monte_carlo_agree_for_weighted_multi_hand_ranges(rank_lookup: RankLookup) -> None:
    players = [
        encode_range("AsAhKdQc@65,QsQhJdTd@35", 4),
        encode_range("KsKhJcTc@40,JsJh8d8c@90", 4),
    ]
    board = encode_board(RIVER)

    exact = compute_equity(players, mode=4, board=board, method="exact", rank_lookup=rank_lookup).equities
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        target_moe=0.0,
        min_samples=20_000,
        max_samples=20_000,
        seed=47,
        method="monte-carlo",
    rank_lookup=rank_lookup).equities

    assert 0 < exact[0] < 1
    assert 0 < exact[1] < 1
    for sampled_equity, exact_equity in zip(
        sampled, exact, strict=True
    ):
        assert sampled_equity == pytest.approx(
            exact_equity, abs=0.015
        )

def test_monte_carlo_flop_agrees_with_exact_equity(rank_lookup: RankLookup) -> None:
    # Integration check: sampled flop equity agrees with exact enumeration.
    # The separate deal_card test verifies that dealing respects the blockers.
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board("2c3d5h")
    exact = compute_equity(players, mode=4, board=board, method="exact", rank_lookup=rank_lookup).equities
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=5_000,
        max_samples=5_000,
        target_moe=0.0,
        seed=53,
        method="monte-carlo",
    rank_lookup=rank_lookup).equities

    for sampled_equity, exact_equity in zip(
        sampled, exact, strict=True
    ):
        assert sampled_equity == pytest.approx(
            exact_equity, abs=0.02
        )


# Input immutability

@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_calculation_does_not_modify_original_ranges(method: str, rank_lookup: RankLookup) -> None:
    players = [
        encode_range("2cAsAhKd,KsKhJdTd", 4),
        encode_range("QsQhJcTc", 4),
    ]
    snapshots = [
        (player.hands.copy(), player.masks.copy(), player.weights.copy())
        for player in players
    ]

    compute_equity(
        players,
        mode=4,
        board=encode_board(RIVER),
        target_moe=0.0,
        min_samples=50,
        max_samples=50,
        seed=41,
        method=method,  # type: ignore[arg-type]
        rank_lookup=rank_lookup,
    )

    for player, (hands, masks, weights) in zip(players, snapshots):
        np.testing.assert_array_equal(player.hands, hands)
        np.testing.assert_array_equal(player.masks, masks)
        np.testing.assert_array_equal(player.weights, weights)
