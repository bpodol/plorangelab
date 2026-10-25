from itertools import combinations, islice

import numpy as np
import pytest

import plorangelab.equity as equity_module
from plorangelab.encoder import (
    EncodedRange,
    build_mask,
    encode_board,
    encode_range,
)
from plorangelab.equity import compute_equity


RIVER = "2c3d5h7s9c"
TURN = "2c3d5h7s"


def _make_range_from_ids(card_rows: list[tuple[int, ...]], mode: int) -> EncodedRange:
    hands = np.asarray(card_rows, dtype=np.uint8).reshape((-1, mode))
    masks = np.asarray([build_mask(hand) for hand in hands], dtype=np.uint64)
    weights = np.ones(len(hands), dtype=np.float32)
    return EncodedRange(hands=hands, masks=masks, weights=weights)


def _configured_ranges(
    player_count: int, mode: int, multi_hand: bool
) -> tuple[list[EncodedRange], np.ndarray]:
    board = encode_board(RIVER)
    blocked = set(int(card) for card in board)
    available = [card for card in range(52) if card not in blocked]
    cards_per_player = mode + 1 if multi_hand else mode
    players = []

    for player_index in range(player_count):
        start = player_index * cards_per_player
        group = available[start : start + cards_per_player]
        hands = [tuple(group[:mode])]
        if multi_hand:
            hands.append(tuple(group[1 : mode + 1]))
        players.append(_make_range_from_ids(hands, mode))

    return players, board


def test_exact_equity_returns_guaranteed_winner_and_loser() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]

    equity = compute_equity(players, mode=4, board=encode_board(RIVER), method="exact")

    assert equity == {"player 1:": 1.0, "player 2:": 0.0}


def test_exact_equity_splits_a_guaranteed_tie_equally() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("AdAcKsQh", 4),
    ]

    equity = compute_equity(players, mode=4, board=encode_board(RIVER), method="exact")

    assert equity["player 1:"] == pytest.approx(0.5)
    assert equity["player 2:"] == pytest.approx(0.5)


def test_exact_equity_splits_a_three_way_tie_into_thirds() -> None:
    board = encode_board("2c3d4hKsQc")
    players = [
        encode_range("As5sKdQd", 4),
        encode_range("Ad5dKhJh", 4),
        encode_range("Ah5hKcJc", 4),
    ]

    equity = compute_equity(players, mode=4, board=board, method="exact")

    for value in equity.values():
        assert value == pytest.approx(1 / 3)


def test_exact_equity_respects_range_weights() -> None:
    players = [
        encode_range("AsAhKdQc@100,QsQhJdTd@50", 4),
        encode_range("KsKhJcTc", 4),
    ]

    equity = compute_equity(players, mode=4, board=encode_board(RIVER), method="exact")

    assert equity["player 1:"] == pytest.approx(2 / 3)
    assert equity["player 2:"] == pytest.approx(1 / 3)


def test_board_blocked_hands_are_removed_before_calculation() -> None:
    players = [
        encode_range("2cAsAhKd,QsQhJdTd", 4),
        encode_range("KsKhJcTc", 4),
    ]

    equity = compute_equity(players, mode=4, board=encode_board(RIVER), method="exact")

    assert equity == {"player 1:": 0.0, "player 2:": 1.0}


@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_colliding_hands_are_never_sent_to_the_evaluator(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    players = [
        encode_range("AsAhKdQc,KsKhJdTd", 4),
        encode_range("AsQsJcTc", 4),
    ]
    original_evaluator = equity_module.numba_evaluate_hands

    def collision_checking_evaluator(hands: np.ndarray, board: np.ndarray) -> np.ndarray:
        masks = np.asarray([build_mask(hand) for hand in hands], dtype=np.uint64)
        assert not equity_module.check_collisions(masks)
        return original_evaluator(hands, board)

    monkeypatch.setattr(equity_module, "numba_evaluate_hands", collision_checking_evaluator)
    equity = compute_equity(
        players,
        mode=4,
        board=encode_board(RIVER),
        samples=30,
        seed=7,
        method=method,  # type: ignore[arg-type]
    )

    assert equity["player 1:"] == pytest.approx(1.0)
    assert equity["player 2:"] == pytest.approx(0.0)


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

    equity = compute_equity(players, mode=mode, board=board, method="exact")

    assert list(equity) == [f"player {index}:" for index in range(1, player_count + 1)]
    assert sum(equity.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "board_text",
    [None, "2c3d5h", "2c3d5h7s", RIVER],
    ids=["preflop", "flop", "turn", "river"],
)
def test_monte_carlo_supports_each_board_state(board_text: str | None) -> None:
    board = None if board_text is None else encode_board(board_text)
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]

    equity = compute_equity(players, mode=4, board=board, samples=8, seed=19)

    assert sum(equity.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_equities_conserve_the_pot(method: str) -> None:
    players, board = _configured_ranges(3, mode=5, multi_hand=True)

    equity = compute_equity(
        players,
        mode=5,
        board=board,
        samples=50,
        seed=23,
        method=method,  # type: ignore[arg-type]
    )

    assert sum(equity.values()) == pytest.approx(1.0)


def test_exact_and_monte_carlo_equity_agree_within_sampling_tolerance() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board(TURN)

    exact = compute_equity(players, mode=4, board=board, method="exact")
    sampled = compute_equity(
        players, mode=4, board=board, samples=5_000, seed=31, method="monte-carlo"
    )

    for player in exact:
        assert sampled[player] == pytest.approx(exact[player], abs=0.01)


def test_exact_and_monte_carlo_agree_for_weighted_multi_hand_ranges() -> None:
    players = [
        encode_range("AsAhKdQc@65,QsQhJdTd@35", 4),
        encode_range("KsKhJcTc@40,JsJh8d8c@90", 4),
    ]
    board = encode_board(RIVER)

    exact = compute_equity(players, mode=4, board=board, method="exact")
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        samples=20_000,
        seed=47,
        method="monte-carlo",
    )

    assert 0 < exact["player 1:"] < 1
    assert 0 < exact["player 2:"] < 1
    for player in exact:
        assert sampled[player] == pytest.approx(exact[player], abs=0.015)


def test_monte_carlo_runout_never_collides_with_selected_hands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    original_evaluator = equity_module.numba_evaluate_hands

    def board_collision_checking_evaluator(
        hands: np.ndarray, board: np.ndarray
    ) -> np.ndarray:
        masks = np.asarray([build_mask(hand) for hand in hands], dtype=np.uint64)
        assert not equity_module.check_collisions(masks)
        assert len(np.unique(board)) == 5
        board_mask = build_mask(board)
        assert all(not (mask & board_mask) for mask in masks)
        return original_evaluator(hands, board)

    monkeypatch.setattr(
        equity_module, "numba_evaluate_hands", board_collision_checking_evaluator
    )
    compute_equity(
        players,
        mode=4,
        board=encode_board("2c3d5h"),
        samples=100,
        seed=53,
        method="monte-carlo",
    )


def test_monte_carlo_results_are_reproducible_for_the_same_seed() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
        encode_range("QsQhJcTc", 4),
    ]
    board = encode_board("2c3d5h")

    first = compute_equity(players, mode=4, board=board, samples=300, seed=101)
    second = compute_equity(players, mode=4, board=board, samples=300, seed=101)

    assert first == second


@pytest.mark.parametrize("player_count", [1, 7])
def test_compute_equity_rejects_player_counts_outside_two_to_six(
    player_count: int,
) -> None:
    player = encode_range("AsAhKdQc", 4)

    with pytest.raises(ValueError):
        compute_equity([player] * player_count, mode=4)


def test_compute_equity_rejects_invalid_plo_mode() -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=6)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "board",
    [
        np.asarray([0, 1], dtype=np.uint8),
        np.asarray([52, 1, 2], dtype=np.uint8),
        np.asarray([0, 0, 2], dtype=np.uint8),
    ],
    ids=["invalid-length", "invalid-card-id", "duplicate-card"],
)
def test_compute_equity_rejects_invalid_boards(board: np.ndarray) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, board=board)


@pytest.mark.parametrize(
    "invalid_range",
    [
        EncodedRange(
            hands=encode_range("AsAhKdQc", 4).hands[0],
            masks=encode_range("AsAhKdQc", 4).masks,
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=encode_range("AsAhKdQc", 4).hands,
            masks=np.asarray([1], dtype=np.uint64),
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=encode_range("AsAhKdQc", 4).hands,
            masks=np.empty((0,), dtype=np.uint64),
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=np.asarray([[51, 51, 45, 42]], dtype=np.uint8),
            masks=encode_range("AsAhKdQc", 4).masks,
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=np.asarray([[51, 50, 45, 42], [51, 50, 45, 42]], dtype=np.uint8),
            masks=np.asarray(
                [build_mask(np.asarray([51, 50, 45, 42], dtype=np.uint8))] * 2,
                dtype=np.uint64,
            ),
            weights=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    ],
    ids=[
        "hand-dimensions",
        "incorrect-mask-content",
        "incorrect-mask-shape",
        "duplicate-cards",
        "duplicate-hands",
    ],
)
def test_compute_equity_rejects_malformed_encoded_ranges(
    invalid_range: EncodedRange,
) -> None:
    players = [invalid_range, encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, board=encode_board(RIVER))


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
    ],
    ids=["nan", "infinite", "negative", "zero-total", "non-float-dtype"],
)
def test_compute_equity_rejects_invalid_weights(weights: np.ndarray) -> None:
    valid = encode_range("AsAhKdQc", 4)
    invalid = EncodedRange(valid.hands, valid.masks, weights)
    players = [invalid, encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, board=encode_board(RIVER))


def test_compute_equity_rejects_zero_weight_hand_in_nonzero_range() -> None:
    encoded = encode_range("AsAhKdQc,KsKhJdTc", 4)
    invalid = EncodedRange(
        encoded.hands,
        encoded.masks,
        np.asarray([0.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="invalid weights"):
        compute_equity([invalid, encode_range("2s3s4s5s", 4)], mode=4)


def test_compute_equity_rejects_empty_ranges() -> None:
    empty = EncodedRange(
        hands=np.empty((0, 4), dtype=np.uint8),
        masks=np.empty((0,), dtype=np.uint64),
        weights=np.empty((0,), dtype=np.float32),
    )
    players = [empty, encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError, match="empty range"):
        compute_equity(players, mode=4)


def test_compute_equity_rejects_ranges_fully_blocked_by_board() -> None:
    players = [
        encode_range("2cAsAhKd", 4),
        encode_range("KsKhJdTd", 4),
    ]

    with pytest.raises(ValueError, match="No combinations left"):
        compute_equity(players, mode=4, board=encode_board(RIVER))


def test_exact_equity_rejects_matchups_with_no_compatible_hands() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("AsQsJcTc", 4),
    ]

    with pytest.raises(ValueError, match="No compatible"):
        compute_equity(players, mode=4, board=encode_board(RIVER), method="exact")


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_compute_equity_rejects_invalid_sample_counts(samples: object) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, samples=samples)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_compute_equity_rejects_invalid_seeds(seed: object) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, seed=seed)  # type: ignore[arg-type]


def test_monte_carlo_raises_when_attempt_limit_is_exceeded() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("AsQsJcTc", 4),
    ]

    with pytest.raises(RuntimeError, match="Sampling limit exceeded"):
        compute_equity(
            players,
            mode=4,
            board=encode_board(RIVER),
            samples=1,
            seed=5,
            method="monte-carlo",
        )


def test_exact_equity_raises_when_workload_limit_is_exceeded() -> None:
    board = encode_board(RIVER)
    board_cards = set(int(card) for card in board)
    available = [card for card in range(52) if card not in board_cards]
    rows = list(islice(combinations(available, 4), 400))
    large_range = _make_range_from_ids(rows, mode=4)

    with pytest.raises(RuntimeError, match="workload limit"):
        compute_equity(
            [large_range, large_range],
            mode=4,
            board=board,
            method="exact",
        )


@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_calculation_does_not_modify_original_ranges(method: str) -> None:
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
        samples=50,
        seed=41,
        method=method,  # type: ignore[arg-type]
    )

    for player, (hands, masks, weights) in zip(players, snapshots):
        np.testing.assert_array_equal(player.hands, hands)
        np.testing.assert_array_equal(player.masks, masks)
        np.testing.assert_array_equal(player.weights, weights)


def test_compute_equity_rejects_invalid_computation_method() -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError, match="Method"):
        compute_equity(
            players,
            mode=4,
            samples=1,
            method="approximate",  # type: ignore[arg-type]
        )
