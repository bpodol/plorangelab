"""Rejection Monte Carlo sampling: legal draws, deterministic results, and batching."""

import numba
import numpy as np
import pytest

import plorangelab.equity.monte_carlo as mc_module
from plorangelab.encoder import build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, TURN

# Legal sampling and board runouts

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


# Determinism and batch-size independence

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
