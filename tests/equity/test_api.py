"""Unified equity public API: validation, dispatch, and cross-engine contracts."""

from itertools import combinations, islice
from threading import Event

import numpy as np
import pytest

import plorangelab.equity.api as equity_module
import plorangelab.equity.monte_carlo as mc_module
from plorangelab.encoder import EncodedRange, build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import (
    RIVER, TURN,
    _configured_ranges, _guaranteed_split_players, _make_range_from_ids,
)


def test_board_blocked_hands_are_removed_before_calculation() -> None:
    players = [
        encode_range("2cAsAhKd,QsQhJdTd", 4),
        encode_range("KsKhJcTc", 4),
    ]

    equity = compute_equity(
        players, mode=4, board=encode_board(RIVER), method="exact"
    ).equities

    assert equity == {"player 1:": 0.0, "player 2:": 1.0}


@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_equities_conserve_the_pot(method: str) -> None:
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
    ).equities

    assert sum(equity.values()) == pytest.approx(1.0)


def test_exact_and_monte_carlo_equity_agree_within_sampling_tolerance() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board(TURN)

    exact = compute_equity(players, mode=4, board=board, method="exact").equities
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=5_000,
        max_samples=5_000,
        target_moe=0.0,
        seed=31,
        method="monte-carlo",
    ).equities

    for player in exact:
        assert sampled[player] == pytest.approx(exact[player], abs=0.01)


def test_exact_and_monte_carlo_agree_for_weighted_multi_hand_ranges() -> None:
    players = [
        encode_range("AsAhKdQc@65,QsQhJdTd@35", 4),
        encode_range("KsKhJcTc@40,JsJh8d8c@90", 4),
    ]
    board = encode_board(RIVER)

    exact = compute_equity(players, mode=4, board=board, method="exact").equities
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        target_moe=0.0,
        min_samples=20_000,
        max_samples=20_000,
        seed=47,
        method="monte-carlo",
    ).equities

    assert 0 < exact["player 1:"] < 1
    assert 0 < exact["player 2:"] < 1
    for player in exact:
        assert sampled[player] == pytest.approx(exact[player], abs=0.015)


def test_monte_carlo_flop_agrees_with_exact_equity() -> None:
    # Integration check: sampled flop equity agrees with exact enumeration.
    # The separate deal_card test verifies that dealing respects the blockers.
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board("2c3d5h")
    exact = compute_equity(players, mode=4, board=board, method="exact").equities
    sampled = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=5_000,
        max_samples=5_000,
        target_moe=0.0,
        seed=53,
        method="monte-carlo",
    ).equities

    for player in exact:
        assert sampled[player] == pytest.approx(exact[player], abs=0.02)


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


@pytest.mark.parametrize("max_samples", [0, -1, 1.5, True])
def test_compute_equity_rejects_invalid_sample_limits(max_samples: object) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(
            players,
            mode=4,
            max_samples=max_samples,  # type: ignore[arg-type]
            method="monte-carlo",
        )


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_compute_equity_rejects_invalid_seeds(seed: object) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, seed=seed)  # type: ignore[arg-type]


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
        target_moe=0.0,
        min_samples=50,
        max_samples=50,
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
            method="approximate",  # type: ignore[arg-type]
        )


def _river_candidate_range(count: int) -> EncodedRange:
    board = encode_board(RIVER)
    available = [card for card in range(52) if card not in set(map(int, board))]
    hands = list(islice(combinations(available, 4), count))
    return _make_range_from_ids(hands, mode=4)


def _stub_monte_carlo_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        equity_module,
        "_run_batched_monte_carlo",
        lambda *args, **kwargs: mc_module.MonteCarloResult(
            equities={"player 1:": 0.75, "player 2:": 0.25},
            margins_of_error={"player 1:": 0.1, "player 2:": 0.1},
            simulations=4,
            attempts=5,
            stop_reason="target_reached",
        ),
    )


def _stub_exact_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        equity_module,
        "run_exact_equity",
        lambda **kwargs: {"player 1:": 1.0, "player 2:": 0.0},
    )


def test_auto_uses_adaptive_monte_carlo_preflop_and_returns_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    _stub_monte_carlo_coordinator(monkeypatch)

    result = compute_equity(players, mode=4, method="auto")

    assert result.method == "monte-carlo"
    assert result.equities == {"player 1:": 0.75, "player 2:": 0.25}
    assert result.margins_of_error == {"player 1:": 0.1, "player 2:": 0.1}
    assert result.simulations == 4
    assert result.attempts == 5
    assert result.stop_reason == "target_reached"


def test_auto_uses_exact_for_a_small_postflop_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = encode_board(RIVER)
    players = [
        _river_candidate_range(1),
        _river_candidate_range(1),
    ]
    _stub_exact_calculation(monkeypatch)

    result = compute_equity(players, mode=4, board=board)

    assert result.method == "exact"
    assert result.equities == {"player 1:": 1.0, "player 2:": 0.0}
    assert result.margins_of_error is None
    assert result.simulations is None
    assert result.attempts is None
    assert result.stop_reason is None


def test_auto_uses_exact_at_the_showdown_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = encode_board(RIVER)
    players = [_river_candidate_range(500), _river_candidate_range(1_000)]
    _stub_exact_calculation(monkeypatch)

    result = compute_equity(players, mode=4, board=board)

    assert result.method == "exact"


def test_auto_uses_monte_carlo_above_the_showdown_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = encode_board(RIVER)
    players = [_river_candidate_range(501), _river_candidate_range(1_000)]
    _stub_monte_carlo_coordinator(monkeypatch)

    result = compute_equity(players, mode=4, board=board)

    assert result.method == "monte-carlo"


def test_auto_estimates_after_removing_board_blocked_hands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = encode_board(RIVER)
    available = [card for card in range(52) if card not in set(map(int, board))]
    blocked_hand = (int(board[0]), *available[:3])
    players = [
        _make_range_from_ids(
            [blocked_hand, *list(islice(combinations(available, 4), 500))],
            mode=4,
        ),
        _river_candidate_range(1_000),
    ]
    captured: dict[str, int] = {}
    original_choose_method = equity_module.choose_method

    def observe_estimate(board_cards: int, estimated_showdowns: int):
        captured["board_cards"] = board_cards
        captured["estimated_showdowns"] = estimated_showdowns
        return original_choose_method(board_cards, estimated_showdowns)

    def exact_after_filtering(**kwargs: object):
        captured["first_range_size"] = len(kwargs["players"][0].hands)
        captured["second_range_size"] = len(kwargs["players"][1].hands)
        return {"player 1:": 1.0, "player 2:": 0.0}

    monkeypatch.setattr(equity_module, "choose_method", observe_estimate)
    monkeypatch.setattr(equity_module, "run_exact_equity", exact_after_filtering)
    result = compute_equity(players, mode=4, board=board)

    assert result.method == "exact"
    assert captured == {
        "board_cards": 5,
        "estimated_showdowns": 500_000,
        "first_range_size": 500,
        "second_range_size": 1_000,
    }


@pytest.mark.parametrize("method", ["exact", "monte-carlo"])
def test_explicit_method_overrides_bypass_auto_selection(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = encode_board(RIVER)
    players = [
        _river_candidate_range(1),
        _river_candidate_range(1),
    ]

    def unexpected_selection(*args: object, **kwargs: object):
        raise AssertionError("explicit methods must bypass automatic selection")

    monkeypatch.setattr(equity_module, "choose_method", unexpected_selection)
    if method == "exact":
        _stub_exact_calculation(monkeypatch)
        result = compute_equity(players, mode=4, board=board, method="exact")
    else:
        _stub_monte_carlo_coordinator(monkeypatch)
        result = compute_equity(
            players,
            mode=4,
            board=board,
            method="monte-carlo",
            min_samples=1,
            max_samples=1,
        )

    assert result.method == method


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_samples": 1_200_001}, "max_samples must not exceed"),
        ({"batch_size": 100_001}, "batch_size must not exceed"),
    ],
)
def test_monte_carlo_rejects_resource_limits(
    kwargs: dict[str, int], message: str
) -> None:
    players, board = _guaranteed_split_players()

    with pytest.raises(ValueError, match=message):
        compute_equity(players, mode=4, board=board, method="monte-carlo", **kwargs)


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
        equity_module.compute_equity(
            players,
            mode=4,
            board=board,
            method="monte-carlo",
            cancel_event=cancelled,
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
        equity_module.compute_equity(
            players,
            mode=4,
            board=board,
            method="monte-carlo",
            cancel_event=cancelled,
        )
