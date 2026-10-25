from itertools import combinations, islice
from threading import Event

import numba
import numpy as np
import pytest
from numba.typed import List

import plorangelab.equity_numba as equity_module
from plorangelab.encoder import (
    EncodedRange,
    build_mask,
    encode_board,
    encode_range,
)
from plorangelab.equity_numba import compute_equity

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
    blocked = {int(card) for card in board}
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


def _run_exact_kernel_equity(kernel, players, mode, board) -> dict[str, float]:
    valid_players = equity_module.prepare_ranges(players, board)
    hands = List([player.hands for player in valid_players])
    masks = List([player.masks for player in valid_players])
    weights = List()
    for player in valid_players:
        normalized = player.weights.astype(np.float64)
        normalized /= normalized.sum(dtype=np.float64)
        weights.append(normalized)

    board_mask = np.uint64(0) if board is None else build_mask(board)
    kernel_args = (hands, masks, weights, board, board_mask, mode)
    if kernel is equity_module._run_exact_equity_kernel_parallel:
        pot_shares, total_weight = kernel(
            *kernel_args, numba.get_num_threads()
        )
    else:
        pot_shares, total_weight = kernel(*kernel_args)
    return {
        f"player {index + 1}:": float(share / total_weight)
        for index, share in enumerate(pot_shares)
    }


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
    )

    assert equity == {
        "player 1:": 0.0,
        "player 2:": 1.0,
    }


def test_monte_carlo_rejects_colliding_hands() -> None:
    # A colliding P1 hand (AA) would win, but its only legal hand loses.
    # This tests rejection through the public API without mocking compiled code.
    players = [
        encode_range("AsAhKdQc,KcQdJhTh", 4),
        encode_range("AsQsJcTc", 4),
    ]

    equity = compute_equity(
        players,
        mode=4,
        board=encode_board(RIVER),
        samples=200,
        seed=7,
        method="monte-carlo",
    )

    assert equity == {"player 1:": 0.0, "player 2:": 1.0}


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


def test_deal_card_never_returns_board_or_hand_cards() -> None:
    # Exercise the same compiled deal_card() used by Monte Carlo directly.
    # We cannot instrument the Numba evaluator with a Python monkeypatch.
    board = encode_board("2c3d5h")
    first = encode_range("AsAhKdQc", 4).hands[0]
    second = encode_range("KsKhJdTd", 4).hands[0]
    blocked = {int(card) for card in np.concatenate((board, first, second))}
    blocked_mask = build_mask(board) | build_mask(first) | build_mask(second)
    deck = np.uint64((1 << 52) - 1) & ~blocked_mask
    rng = np.random.default_rng(53)
    dealt: set[int] = set()

    for _ in range(52 - len(blocked)):
        card, deck = equity_module.deal_card(deck, rng)
        assert card not in blocked
        assert card not in dealt
        assert not (deck & (np.uint64(1) << np.uint64(card)))
        dealt.add(card)

    assert dealt == set(range(52)) - blocked
    assert deck == 0
    with pytest.raises(ValueError, match="Deck is empty"):
        equity_module.deal_card(deck, rng)


def test_monte_carlo_flop_agrees_with_exact_equity() -> None:
    # Integration check: sampled flop equity agrees with exact enumeration.
    # The separate deal_card test verifies that dealing respects the blockers.
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board("2c3d5h")
    exact = compute_equity(players, mode=4, board=board, method="exact")
    sampled = compute_equity(
        players, mode=4, board=board, samples=5_000, seed=53,
        method="monte-carlo",
    )

    for player in exact:
        assert sampled[player] == pytest.approx(exact[player], abs=0.02)


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
    board_cards = {int(card) for card in board}
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

@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_exact_equity_is_invariant_to_weight_scaling(scale):
    players = [
        encode_range("AsAhKdQc@70,QsQhJdTd@30", 4),
        encode_range("KsKhJcTc@40,JsJh8d8c@60", 4),
    ]

    board = encode_board("2c3d5h7s9c")

    baseline = compute_equity(
        players, mode=4, board=board, method="exact"
    )

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
    )

    for player in baseline:
        assert actual[player] == pytest.approx(
            baseline[player], abs=1e-12
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


def test_monte_carlo_deterministic_across_thread_counts() -> None:
    """Same seed must produce identical equities with 1, 2, 4, or 8 threads."""

    players = [
        encode_range("AsKsQhJhTh@75,AdKdQcJcTc@25", 5),
        encode_range("As9h8s8h7d@40,6c6d5c5d4h@60", 5),
    ]
    board = encode_board("2c3d4s")

    original_threads = numba.get_num_threads()
    results = []
    thread_counts = [
        n for n in (1, 2, 4, 8)
        if n <= numba.config.NUMBA_NUM_THREADS
    ]

    if len(thread_counts) < 2:
        pytest.skip("At least 2 Numba threads required")

    try:
        for threads in thread_counts:
            numba.set_num_threads(threads)

            equity = compute_equity(
                players=players,
                mode=5,
                board=board,
                samples=10_000,
                seed=42,
                method="monte-carlo",
            )
            results.append(equity)

        for result in results[1:]:
            assert result == results[0]

    finally:
        numba.set_num_threads(original_threads)


def _guaranteed_split_players() -> tuple[list[EncodedRange], np.ndarray]:
    board = encode_board("3d4h5s9cTc")
    players = [
        encode_range("6c7cAsAh", 4),
        encode_range("6d7dKsKh", 4),
    ]
    return players, board


def test_adaptive_monte_carlo_stops_when_target_is_reached() -> None:
    players, board = _guaranteed_split_players()

    result = equity_module.compute_equity_adaptive(
        players,
        mode=4,
        board=board,
        target_moe=0.001,
        min_samples=4,
        max_samples=12,
        batch_size=4,
        seed=13,
    )

    assert result.stop_reason == "target_reached"
    assert result.simulations == 4
    assert result.attempts == 4
    assert result.equities == {"player 1:": 0.5, "player 2:": 0.5}
    assert result.margins_of_error == {"player 1:": 0.0, "player 2:": 0.0}


def test_adaptive_stops_at_max_samples_when_precision_is_not_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(equity_module, "_run_monte_carlo_batch", varying_batch)
    result = equity_module.compute_equity_adaptive(
        players,
        mode=4,
        board=board,
        target_moe=0.0,
        min_samples=2,
        max_samples=5,
        batch_size=2,
        seed=13,
    )

    assert result.stop_reason == "max_simulations"
    assert result.simulations == 5
    assert result.attempts == 5
    assert result.margins_of_error["player 1:"] > 0
    assert result.margins_of_error["player 2:"] > 0


def test_equity_statistics_include_split_pots_and_sample_variance() -> None:
    # Observations are (1, 0), (0.5, 0.5), and (0, 1): one split pot.
    equities, margins = equity_module._equity_statistics(
        np.asarray([1.5, 1.5]),
        np.asarray([1.25, 1.25]),
        simulations=3,
        confidence_multiplier=1.96,
    )

    expected_moe = 1.96 * np.sqrt(0.25 / 3)
    assert equities == {"player 1:": 0.5, "player 2:": 0.5}
    assert margins["player 1:"] == pytest.approx(expected_moe)
    assert margins["player 2:"] == pytest.approx(expected_moe)


def test_adaptive_cancellation_before_execution_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    cancelled.set()

    def unexpected_batch(**kwargs: object):
        raise AssertionError("the batch kernel must not run after pre-cancellation")

    monkeypatch.setattr(equity_module, "_run_monte_carlo_batch", unexpected_batch)
    with pytest.raises(InterruptedError, match="before any simulations"):
        equity_module.compute_equity_adaptive(
            players,
            mode=4,
            board=board,
            min_samples=2,
            max_samples=4,
            batch_size=2,
            cancel_event=cancelled,
        )


def test_adaptive_cancellation_between_batches_returns_partial_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(equity_module, "_run_monte_carlo_batch", tied_batch)
    result = equity_module.compute_equity_adaptive(
        players,
        mode=4,
        board=board,
        min_samples=4,
        max_samples=8,
        batch_size=2,
        cancel_event=cancelled,
    )

    assert calls == 1
    assert result.stop_reason == "cancelled"
    assert result.simulations == 2
    assert result.equities == {"player 1:": 0.5, "player 2:": 0.5}


def test_attempt_limit_returns_partial_adaptive_batch_without_losing_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(equity_module, "_run_monte_carlo_batch", exhausting_batch)
    result = equity_module.compute_equity_adaptive(
        players,
        mode=4,
        board=board,
        min_samples=1,
        max_samples=4,
        batch_size=2,
    )

    assert result.stop_reason == "attempt_limit"
    assert result.simulations == 1
    assert result.attempts == 10_000
    assert result.equities == {"player 1:": 1.0, "player 2:": 0.0}


def test_fixed_sample_seeded_result_matches_legacy_and_is_batch_independent() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board(TURN)

    small_batches = compute_equity(
        players, mode=4, board=board, samples=5_000, seed=31, batch_size=127
    )
    single_batch = compute_equity(
        players, mode=4, board=board, samples=5_000, seed=31, batch_size=10_000
    )

    assert small_batches == single_batch
    assert small_batches == {"player 1:": 0.9762, "player 2:": 0.0238}


def test_fixed_sample_memory_is_bounded_by_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _configured_ranges(2, mode=4, multi_hand=False)
    original_batch = equity_module._run_monte_carlo_batch
    allocated_batch_sizes: list[int] = []

    def recording_batch(**kwargs: object):
        allocated_batch_sizes.append(int(kwargs["batch_size"]))
        return original_batch(**kwargs)

    monkeypatch.setattr(equity_module, "_run_monte_carlo_batch", recording_batch)
    result = compute_equity(
        players, mode=4, board=board, samples=11, seed=17, batch_size=4
    )

    assert sum(allocated_batch_sizes) == 11
    assert allocated_batch_sizes == [4, 4, 3]
    assert max(allocated_batch_sizes) <= 4
    assert sum(result.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("fixed", {"samples": 1_000_001}, "samples must not exceed"),
        ("fixed", {"batch_size": 100_001}, "batch_size must not exceed"),
        ("adaptive", {"max_samples": 1_000_001}, "max_samples must not exceed"),
        ("adaptive", {"batch_size": 100_001}, "batch_size must not exceed"),
    ],
)
def test_monte_carlo_rejects_resource_limits(
    call: str, kwargs: dict[str, int], message: str
) -> None:
    players, board = _guaranteed_split_players()

    with pytest.raises(ValueError, match=message):
        if call == "fixed":
            compute_equity(players, mode=4, board=board, **kwargs)
        else:
            equity_module.compute_equity_adaptive(
                players, mode=4, board=board, **kwargs
            )


def test_adaptive_cancellation_precedes_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    cancelled.set()

    def unexpected_validation(*args: object, **kwargs: object) -> None:
        raise AssertionError("cancelled work must skip input validation")

    monkeypatch.setattr(
        equity_module, "validate_equity_inputs", unexpected_validation
    )
    with pytest.raises(InterruptedError, match="before validation or preparation"):
        equity_module.compute_equity_adaptive(
            players, mode=4, board=board, cancel_event=cancelled
        )


def test_adaptive_cancellation_after_validation_skips_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()

    def cancel_during_validation(*args: object, **kwargs: object) -> None:
        cancelled.set()

    def unexpected_preparation(*args: object, **kwargs: object):
        raise AssertionError("cancelled work must skip range preparation")

    monkeypatch.setattr(
        equity_module, "validate_equity_inputs", cancel_during_validation
    )
    monkeypatch.setattr(
        equity_module, "_prepare_monte_carlo", unexpected_preparation
    )
    with pytest.raises(InterruptedError, match="before preparation"):
        equity_module.compute_equity_adaptive(
            players, mode=4, board=board, cancel_event=cancelled
        )


def test_attempts_are_capped_per_batch_and_global_attempts_accumulate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    calls = 0

    def capped_batch(**kwargs: object):
        nonlocal calls
        calls += 1
        assert kwargs["attempt_budget"] == equity_module._MAX_BATCH_ATTEMPTS
        if calls == 1:
            return (
                np.asarray([0.0, 0.0]),
                np.asarray([0.0, 0.0]),
                0,
                equity_module._MAX_BATCH_ATTEMPTS,
            )
        cancelled.set()
        return (
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            1,
            1,
        )

    monkeypatch.setattr(equity_module, "_run_monte_carlo_batch", capped_batch)
    result = equity_module.compute_equity_adaptive(
        players,
        mode=4,
        board=board,
        target_moe=0.0,
        min_samples=1,
        max_samples=20_000,
        batch_size=10_000,
        cancel_event=cancelled,
    )

    assert calls == 2
    assert result.stop_reason == "cancelled"
    assert result.simulations == 1
    assert result.attempts == equity_module._MAX_BATCH_ATTEMPTS + 1
    assert result.equities == {"player 1:": 1.0, "player 2:": 0.0}


def test_fixed_sample_results_are_reproducible_across_batch_sizes_with_three_way_ties(
) -> None:
    board = encode_board("3d4h5s9cTc")
    players = [
        encode_range("6c7cAsAh", 4),
        encode_range("6d7dKsKh", 4),
        encode_range("6h7hQsQd", 4),
    ]
    results = [
        compute_equity(
            players,
            mode=4,
            board=board,
            samples=101,
            seed=23,
            batch_size=batch_size,
        )
        for batch_size in (7, 31, 101)
    ]

    for result in results[1:]:
        np.testing.assert_allclose(
            list(result.values()), list(results[0].values()), rtol=0, atol=1e-15
        )
    assert list(results[0].values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize("board_cards", [3, 4, 5])
@pytest.mark.parametrize("player_count", [2, 3, 4, 5, 6])
def test_parallel_exact_matches_sequential_across_modes_streets_and_players(
    mode: int,
    board_cards: int,
    player_count: int,
) -> None:
    board = np.arange(board_cards, dtype=np.uint8)
    blocked = set(int(card) for card in board)
    available = [card for card in range(52) if card not in blocked]
    players = []
    for player_index in range(player_count):
        start = player_index * mode
        players.append(
            _make_range_from_ids(
                [tuple(available[start : start + mode])], mode
            )
        )

    sequential = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel, players, mode, board
    )
    parallel = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel_parallel, players, mode, board
    )

    np.testing.assert_allclose(
        list(parallel.values()), list(sequential.values()), rtol=0, atol=1e-12
    )
    assert sum(parallel.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", [4, 5])
def test_parallel_exact_matches_weighted_ranges_with_collisions(mode: int) -> None:
    board = encode_board(RIVER)
    if mode == 4:
        players = [
            encode_range("AsAhKdQc@100,KsKhQdJc@35", mode),
            encode_range("AsQsJcTc@50,7c8c9dTd@100", mode),
        ]
    else:
        players = [
            encode_range("AsAhKdQcJh@100,KsKhQdJcTc@35", mode),
            encode_range("AsQsJcTc9h@50,7c8c9dTdJh@100", mode),
        ]

    sequential = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel, players, mode, board
    )
    parallel = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel_parallel, players, mode, board
    )

    np.testing.assert_allclose(
        list(parallel.values()), list(sequential.values()), rtol=0, atol=1e-12
    )
    assert sum(parallel.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", [4, 5])
@pytest.mark.parametrize("player_count", [3, 4, 5, 6])
def test_parallel_exact_multiway_weighted_collisions_use_multiple_partitions(
    mode: int, player_count: int
) -> None:
    board = encode_board(RIVER)
    blocked = {int(card) for card in board}
    available = [card for card in range(52) if card not in blocked]
    primary_cards = available[: player_count * mode]
    players = []

    for player_index in range(player_count):
        primary_start = player_index * mode
        primary = tuple(primary_cards[primary_start : primary_start + mode])
        alternate_start = (primary_start + 1) % len(primary_cards)
        alternate = tuple(
            primary_cards[(alternate_start + offset) % len(primary_cards)]
            for offset in range(mode)
        )
        player = _make_range_from_ids([primary, alternate], mode)
        player.weights[:] = np.asarray(
            [0.2 + 0.05 * player_index, 0.8 - 0.05 * player_index],
            dtype=np.float32,
        )
        players.append(player)

    assert any(
        ((players[first].masks[:, None] & players[second].masks[None, :]) != 0).any()
        for first in range(player_count)
        for second in range(first + 1, player_count)
    )
    matchup_count = 2**player_count
    original_threads = numba.get_num_threads()
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("At least 2 Numba threads are required for parallel partitions")

    try:
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        assert matchup_count >= numba.get_num_threads()
        sequential = _run_exact_kernel_equity(
            equity_module._run_exact_equity_kernel, players, mode, board
        )
        parallel = _run_exact_kernel_equity(
            equity_module._run_exact_equity_kernel_parallel, players, mode, board
        )
    finally:
        numba.set_num_threads(original_threads)

    np.testing.assert_allclose(
        list(parallel.values()), list(sequential.values()), rtol=0, atol=1e-12
    )
    assert sum(parallel.values()) == pytest.approx(1.0)


def test_exact_public_api_uses_sequential_kernel_with_one_numba_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = encode_board(RIVER)
    blocked = {int(card) for card in board}
    available = [card for card in range(52) if card not in blocked]
    pool_size = len(available) // 2
    players = [
        _make_range_from_ids(
            list(
                islice(
                    combinations(
                        available[index * pool_size : (index + 1) * pool_size],
                        4,
                    ),
                    150,
                )
            ),
            4,
        )
        for index in range(2)
    ]

    def parallel_must_not_run(*args, **kwargs):
        raise AssertionError("parallel kernel should not run with one thread")

    monkeypatch.setattr(equity_module, "get_num_threads", lambda: 1)
    monkeypatch.setattr(
        equity_module,
        "_run_exact_equity_kernel_parallel",
        parallel_must_not_run,
    )
    result = compute_equity(players, mode=4, board=board, method="exact")
    assert sum(result.values()) == pytest.approx(1.0)


def test_parallel_exact_preserves_multiway_split_pots() -> None:
    board = encode_board("2c3d4hKsQc")
    players = [
        encode_range("As5sKdQd", 4),
        encode_range("Ad5dKhJh", 4),
        encode_range("Ah5hKcJc", 4),
    ]

    sequential = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel, players, 4, board
    )
    parallel = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel_parallel, players, 4, board
    )

    assert list(sequential.values()) == pytest.approx([1 / 3] * 3)
    assert list(parallel.values()) == pytest.approx([1 / 3] * 3)
    np.testing.assert_allclose(
        list(parallel.values()), list(sequential.values()), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("mode", [4, 5])
def test_parallel_exact_handles_one_by_thousand_unbalanced_ranges(mode: int) -> None:
    board = encode_board(RIVER)
    blocked = set(int(card) for card in board)
    available = [card for card in range(52) if card not in blocked]
    pool_size = len(available) // 2
    first_pool = available[:pool_size]
    second_pool = available[pool_size : 2 * pool_size]
    first = _make_range_from_ids([tuple(first_pool[:mode])], mode)
    second_hands = list(islice(combinations(second_pool, mode), 1_000))
    second = _make_range_from_ids(second_hands, mode)
    players = [first, second]

    sequential = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel, players, mode, board
    )
    parallel = _run_exact_kernel_equity(
        equity_module._run_exact_equity_kernel_parallel, players, mode, board
    )

    np.testing.assert_allclose(
        list(parallel.values()), list(sequential.values()), rtol=0, atol=1e-12
    )
    assert sum(parallel.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", [4, 5])
def test_parallel_exact_is_reproducible_across_thread_counts(mode: int) -> None:
    board = encode_board(RIVER)
    blocked = set(int(card) for card in board)
    available = [card for card in range(52) if card not in blocked]
    pool_size = len(available) // 2
    players = []
    for player_index in range(2):
        pool = available[player_index * pool_size : (player_index + 1) * pool_size]
        hands = list(islice(combinations(pool, mode), 200))
        player = _make_range_from_ids(hands, mode)
        if player_index == 0:
            player.weights[:] = np.linspace(0.1, 1.0, 200, dtype=np.float32)
        else:
            player.weights[:] = np.linspace(1.0, 0.1, 200, dtype=np.float32)
        players.append(player)

    original_threads = numba.get_num_threads()
    thread_counts = [
        count for count in (1, 2, 4, 8)
        if count <= numba.config.NUMBA_NUM_THREADS
    ]
    if len(thread_counts) < 2:
        pytest.skip("At least 2 Numba threads required")

    results = []
    try:
        for count in thread_counts:
            numba.set_num_threads(count)
            first = compute_equity(players, mode=mode, board=board, method="exact")
            repeated = compute_equity(
                players, mode=mode, board=board, method="exact"
            )
            assert repeated == first
            results.append(list(first.values()))
    finally:
        numba.set_num_threads(original_threads)

    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=0, atol=1e-12)
