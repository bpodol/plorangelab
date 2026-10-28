"""Exact enumeration: winning, tied, weighted, colliding, and bounded matchups."""

from itertools import combinations, islice

import numpy as np
import pytest

from plorangelab.encoder import EncodedRange, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, _configured_ranges, _make_range_from_ids

# Equity correctness

def test_exact_equity_returns_guaranteed_winner_and_loser() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]

    equity = compute_equity(
        players, mode=4, board=encode_board(RIVER), method="exact"
    ).equities

    assert equity == {"player 1:": 1.0, "player 2:": 0.0}

def test_exact_equity_splits_a_guaranteed_tie_equally() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("AdAcKsQh", 4),
    ]

    equity = compute_equity(
        players, mode=4, board=encode_board(RIVER), method="exact"
    ).equities

    assert equity["player 1:"] == pytest.approx(0.5)
    assert equity["player 2:"] == pytest.approx(0.5)

def test_exact_equity_splits_a_three_way_tie_into_thirds() -> None:
    board = encode_board("2c3d4hKsQc")
    players = [
        encode_range("As5sKdQd", 4),
        encode_range("Ad5dKhJh", 4),
        encode_range("Ah5hKcJc", 4),
    ]

    equity = compute_equity(players, mode=4, board=board, method="exact").equities

    for value in equity.values():
        assert value == pytest.approx(1 / 3)

def test_exact_equity_respects_range_weights() -> None:
    players = [
        encode_range("AsAhKdQc@100,QsQhJdTd@50", 4),
        encode_range("KsKhJcTc", 4),
    ]

    equity = compute_equity(
        players, mode=4, board=encode_board(RIVER), method="exact"
    ).equities

    assert equity["player 1:"] == pytest.approx(2 / 3)
    assert equity["player 2:"] == pytest.approx(1 / 3)

def test_exact_equity_excludes_colliding_hands() -> None:
    # P1's first hand collides with P2 and would win.
    # P1's only compatible hand loses.

    players = [
        encode_range("AsAhKdQc,KcQdJhTh", 4),
        encode_range("AsQsJcTc", 4),
    ]

    equity = compute_equity(
        players,
        mode=4,
        board=encode_board(RIVER),
        method="exact",
    ).equities

    assert equity == {
        "player 1:": 0.0,
        "player 2:": 1.0,
    }

@pytest.mark.parametrize(
    ("player_count", "mode", "multi_hand"),
    [
        (player_count, mode, multi_hand)
        for player_count in (2, 3, 6)
        for mode in (4, 5)
        for multi_hand in (False, True)
    ],
)
def test_exact_equity_supports_player_modes_and_range_sizes(
    player_count: int, mode: int, multi_hand: bool
) -> None:
    players, board = _configured_ranges(player_count, mode, multi_hand)

    equity = compute_equity(players, mode=mode, board=board, method="exact").equities

    assert list(equity) == [f"player {index}:" for index in range(1, player_count + 1)]
    assert sum(equity.values()) == pytest.approx(1.0)


# Limits and invariance

def test_exact_equity_rejects_matchups_with_no_compatible_hands() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("AsQsJcTc", 4),
    ]

    with pytest.raises(ValueError, match="No compatible"):
        compute_equity(players, mode=4, board=encode_board(RIVER), method="exact")

def test_exact_equity_raises_when_workload_limit_is_exceeded() -> None:
    board = encode_board(RIVER)
    board_cards = {int(card) for card in board}
    available = [card for card in range(52) if card not in board_cards]
    rows = list(islice(combinations(available, 4), 708))
    large_range = _make_range_from_ids(rows, mode=4)

    with pytest.raises(RuntimeError, match="workload limit"):
        compute_equity(
            [large_range, large_range],
            mode=4,
            board=board,
            method="exact",
        )

@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_exact_equity_is_invariant_to_weight_scaling(scale):
    players = [
        encode_range("AsAhKdQc@70,QsQhJdTd@30", 4),
        encode_range("KsKhJcTc@40,JsJh8d8c@60", 4),
    ]

    board = encode_board("2c3d5h7s9c")

    baseline = compute_equity(players, mode=4, board=board, method="exact").equities

    scaled_players = [
        EncodedRange(
            hands=p.hands,
            masks=p.masks,
            weights=p.weights.astype(np.float64) * scale,
        )
        for p in players
    ]

    actual = compute_equity(
        scaled_players, mode=4, board=board, method="exact"
    ).equities

    for player in baseline:
        assert actual[player] == pytest.approx(
            baseline[player], abs=1e-12
        )
