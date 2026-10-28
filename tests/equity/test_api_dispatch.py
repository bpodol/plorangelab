"""Public equity API orchestration: method selection, diagnostics, and cancellation."""

from itertools import combinations, islice
from threading import Event

import pytest

import plorangelab.equity.engine as equity_module
import plorangelab.equity.monte_carlo as mc_module
from plorangelab.encoder import EncodedRange, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, _guaranteed_split_players, _make_range_from_ids


# Local dispatch test doubles

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


# Automatic and explicit dispatch

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


# Cancellation lifecycle before computation

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
