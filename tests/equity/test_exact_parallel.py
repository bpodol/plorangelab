"""Parallel exact enumeration must agree with sequential execution across workloads."""

from itertools import combinations, islice

import numba
import numpy as np
import pytest
from numba.typed import List

import plorangelab.equity.api as equity_module
import plorangelab.equity.exact as exact_module
from plorangelab.encoder import build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, _make_range_from_ids


# Sequential/parallel comparison helper

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
    if kernel is exact_module._run_exact_equity_kernel_parallel:
        pot_shares, total_weight = kernel(
            *kernel_args, numba.get_num_threads()
        )
    else:
        pot_shares, total_weight = kernel(*kernel_args)
    return {
        f"player {index + 1}:": float(share / total_weight)
        for index, share in enumerate(pot_shares)
    }


# Parallel correctness and workload shapes

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
        exact_module._run_exact_equity_kernel, players, mode, board
    )
    parallel = _run_exact_kernel_equity(
        exact_module._run_exact_equity_kernel_parallel, players, mode, board
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
        exact_module._run_exact_equity_kernel, players, mode, board
    )
    parallel = _run_exact_kernel_equity(
        exact_module._run_exact_equity_kernel_parallel, players, mode, board
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
            exact_module._run_exact_equity_kernel, players, mode, board
        )
        parallel = _run_exact_kernel_equity(
            exact_module._run_exact_equity_kernel_parallel, players, mode, board
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

    monkeypatch.setattr(exact_module, "get_num_threads", lambda: 1)
    monkeypatch.setattr(
        exact_module,
        "_run_exact_equity_kernel_parallel",
        parallel_must_not_run,
    )
    result = compute_equity(players, mode=4, board=board, method="exact")
    assert sum(result.equities.values()) == pytest.approx(1.0)

def test_parallel_exact_preserves_multiway_split_pots() -> None:
    board = encode_board("2c3d4hKsQc")
    players = [
        encode_range("As5sKdQd", 4),
        encode_range("Ad5dKhJh", 4),
        encode_range("Ah5hKcJc", 4),
    ]

    sequential = _run_exact_kernel_equity(
        exact_module._run_exact_equity_kernel, players, 4, board
    )
    parallel = _run_exact_kernel_equity(
        exact_module._run_exact_equity_kernel_parallel, players, 4, board
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
        exact_module._run_exact_equity_kernel, players, mode, board
    )
    parallel = _run_exact_kernel_equity(
        exact_module._run_exact_equity_kernel_parallel, players, mode, board
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
            results.append(list(first.equities.values()))
    finally:
        numba.set_num_threads(original_threads)

    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=0, atol=1e-12)
