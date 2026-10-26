"""Adaptive Monte Carlo: sampling, statistics, limits, and cancellation."""

from threading import Event

import numba
import numpy as np
import pytest

import plorangelab.equity.api as equity_module
import plorangelab.equity.monte_carlo as mc_module
from plorangelab.encoder import build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, TURN, _configured_ranges, _guaranteed_split_players


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
        target_moe=0.0,
        min_samples=200,
        max_samples=200,
        seed=7,
        method="monte-carlo",
    ).equities

    assert equity == {"player 1:": 0.0, "player 2:": 1.0}


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

    equity = compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=8,
        max_samples=8,
        seed=19,
    ).equities

    assert sum(equity.values()) == pytest.approx(1.0)


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
        card, deck = mc_module.deal_card(deck, rng)
        assert card not in blocked
        assert card not in dealt
        assert not (deck & (np.uint64(1) << np.uint64(card)))
        dealt.add(card)

    assert dealt == set(range(52)) - blocked
    assert deck == 0
    with pytest.raises(ValueError, match="Deck is empty"):
        mc_module.deal_card(deck, rng)


def test_monte_carlo_results_are_reproducible_for_the_same_seed() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
        encode_range("QsQhJcTc", 4),
    ]
    board = encode_board("2c3d5h")

    first = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=300,
        max_samples=300,
        target_moe=0.0,
        seed=101,
        method="monte-carlo",
    ).equities
    second = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=300,
        max_samples=300,
        target_moe=0.0,
        seed=101,
        method="monte-carlo",
    ).equities

    assert first == second


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
            min_samples=1,
            max_samples=1,
            seed=5,
            method="monte-carlo",
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
                min_samples=10_000,
                max_samples=10_000,
                target_moe=0.0,
                seed=42,
                method="monte-carlo",
            )
            results.append(equity)

        for result in results[1:]:
            assert result == results[0]

    finally:
        numba.set_num_threads(original_threads)


def test_adaptive_monte_carlo_stops_when_target_is_reached() -> None:
    players, board = _guaranteed_split_players()

    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        target_moe=0.001,
        min_samples=4,
        max_samples=12,
        batch_size=4,
        seed=13,
    )

    assert result.method == "monte-carlo"
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

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", varying_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
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
    equities, margins = mc_module._equity_statistics(
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

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", unexpected_batch)
    with pytest.raises(InterruptedError, match="before any simulations"):
        equity_module.compute_equity(
            players,
            mode=4,
            board=board,
            method="monte-carlo",
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

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", tied_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
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

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", exhausting_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        min_samples=1,
        max_samples=4,
        batch_size=2,
    )

    assert result.stop_reason == "attempt_limit"
    assert result.simulations == 1
    assert result.attempts == 10_000
    assert result.equities == {"player 1:": 1.0, "player 2:": 0.0}


def test_fixed_budget_seeded_result_matches_legacy_and_is_batch_independent() -> None:
    players = [
        encode_range("AsAhKdQc", 4),
        encode_range("KsKhJdTd", 4),
    ]
    board = encode_board(TURN)

    small_batches = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=5_000,
        max_samples=5_000,
        target_moe=0.0,
        seed=31,
        method="monte-carlo",
        batch_size=127,
    ).equities
    single_batch = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=5_000,
        max_samples=5_000,
        target_moe=0.0,
        seed=31,
        method="monte-carlo",
        batch_size=10_000,
    ).equities

    assert small_batches == single_batch
    assert small_batches == {"player 1:": 0.9762, "player 2:": 0.0238}


def test_fixed_budget_memory_is_bounded_by_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _configured_ranges(2, mode=4, multi_hand=False)
    original_batch = mc_module._run_monte_carlo_batch
    allocated_batch_sizes: list[int] = []

    def recording_batch(**kwargs: object):
        allocated_batch_sizes.append(int(kwargs["batch_size"]))
        return original_batch(**kwargs)

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", recording_batch)
    result = compute_equity(
        players,
        mode=4,
        board=board,
        min_samples=11,
        max_samples=11,
        target_moe=0.0,
        seed=17,
        method="monte-carlo",
        batch_size=4,
    )

    assert sum(allocated_batch_sizes) == 11
    assert allocated_batch_sizes == [4, 4, 3]
    assert max(allocated_batch_sizes) <= 4
    assert sum(result.equities.values()) == pytest.approx(1.0)


def test_attempts_are_capped_per_batch_and_global_attempts_accumulate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    players, board = _guaranteed_split_players()
    cancelled = Event()
    calls = 0

    def capped_batch(**kwargs: object):
        nonlocal calls
        calls += 1
        assert kwargs["attempt_budget"] == mc_module._MAX_BATCH_ATTEMPTS
        if calls == 1:
            return (
                np.asarray([0.0, 0.0]),
                np.asarray([0.0, 0.0]),
                0,
                mc_module._MAX_BATCH_ATTEMPTS,
            )
        cancelled.set()
        return (
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            1,
            1,
        )

    monkeypatch.setattr(mc_module, "_run_monte_carlo_batch", capped_batch)
    result = equity_module.compute_equity(
        players,
        mode=4,
        board=board,
        method="monte-carlo",
        target_moe=0.0,
        min_samples=1,
        max_samples=20_000,
        batch_size=10_000,
        cancel_event=cancelled,
    )

    assert calls == 2
    assert result.stop_reason == "cancelled"
    assert result.simulations == 1
    assert result.attempts == mc_module._MAX_BATCH_ATTEMPTS + 1
    assert result.equities == {"player 1:": 1.0, "player 2:": 0.0}


def test_fixed_budget_three_way_ties_are_batch_independent() -> None:
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
            target_moe=0.0,
            min_samples=101,
            max_samples=101,
            seed=23,
            method="monte-carlo",
            batch_size=batch_size,
        ).equities
        for batch_size in (7, 31, 101)
    ]

    for result in results[1:]:
        np.testing.assert_allclose(
            list(result.values()), list(results[0].values()), rtol=0, atol=1e-15
        )
    assert list(results[0].values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3])



def test_importance_sampler_agrees_with_exact_weighted_colliding_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plorangelab.equity import api as api_module

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
    from plorangelab.equity.importance_sampling import _importance_statistics

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