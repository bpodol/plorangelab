"""Exact PLO equity enumeration and sequential/parallel Numba dispatch."""

from math import comb, prod

import numpy as np
from numba import get_num_threads, njit, prange
from numba.typed import List
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange, build_mask
from plorangelab.rank_combination import RankLookup
from plorangelab.types import Mode

from .evaluator import numba_evaluate_hands

_EXACT_PARALLEL_MIN_SHOWDOWNS = 1_000
_EXACT_MAX_SHOWDOWNS = 500_000


@njit(inline="always")
def _accumulate_exact_showdown(
    selected_hands,
    current_board,
    player_count,
    matchup_weight,
    pot_shares,
    total_weight,
    rank_table,
    binomial_table,
):
    """Evaluate one showdown and add its weighted pot shares."""
    scores = numba_evaluate_hands(selected_hands, current_board, rank_table=rank_table, binomial_table=binomial_table)

    highest_score = scores[0]
    for p in range(1, player_count):
        highest_score = max(highest_score, scores[p])

    winners_count = 0
    for p in range(player_count):
        if scores[p] == highest_score:
            winners_count += 1

    share = matchup_weight / winners_count
    for p in range(player_count):
        if scores[p] == highest_score:
            pot_shares[p] += share

    return total_weight + matchup_weight



@njit(cache=True)
def _run_exact_equity_kernel(
    hands,
    masks,
    weights,
    board,
    board_mask,
    mode,
    rank_table,
    binomial_table,
):
    """
    Enumerate all compatible matchups and board runouts. Complete river boards
    are evaluated directly without constructing a remaining-deck runout.

    Args:
        hands: Numba typed list of uint8 hand arrays.
        masks: Numba typed list of uint64 mask arrays.
        weights: Numba typed list of normalized float64 weights.
        board: Known board cards, or None.
        board_mask: Bitmask of known board cards.
        mode: 4 or 5.

    Returns:
        (weighted_pot_shares, total_weight)
    """
    player_count = len(hands)
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len

    selected_hands = np.empty(
        (player_count, mode), dtype=np.uint8
    )
    current_board = np.empty(5, dtype=np.uint8)

    if board is not None:
        for j in range(board_len):
            current_board[j] = board[j]

    # Reusable runout buffers stay empty for a complete river board.
    available = np.empty(52 if missing_board else 0, dtype=np.uint8)
    runout_indices = np.empty(missing_board, dtype=np.int64)
    hand_indices = np.zeros(player_count, dtype=np.int64)

    pot_shares = np.zeros(player_count, dtype=np.float64)
    total_weight = 0.0

    # Enumerate the Cartesian product of player ranges.
    while True:
        used_mask = board_mask
        matchup_weight = 1.0
        valid = True

        for p in range(player_count):
            index = hand_indices[p]
            hand_mask = masks[p][index]

            if used_mask & hand_mask:
                valid = False
                break

            used_mask |= hand_mask
            matchup_weight *= weights[p][index]

            if matchup_weight == 0.0:
                valid = False
                break

            for c in range(mode):
                selected_hands[p, c] = hands[p][index, c]

        if valid:
            if missing_board == 0:
                total_weight = _accumulate_exact_showdown(
                    selected_hands,
                    current_board,
                    player_count,
                    matchup_weight,
                    pot_shares,
                    total_weight,
                    rank_table=rank_table,
                    binomial_table=binomial_table
                )
            else:
                # Build the deck remaining after blockers.
                available_count = 0

                for card in range(52):
                    bit = np.uint64(1) << np.uint64(card)

                    if (used_mask & bit) == 0:
                        available[available_count] = card
                        available_count += 1

                # Initialize lexicographic combination indices.
                for j in range(missing_board):
                    runout_indices[j] = j

                # Enumerate every possible board completion.
                while True:
                    for j in range(missing_board):
                        current_board[board_len + j] = (
                            available[runout_indices[j]]
                        )

                    total_weight = _accumulate_exact_showdown(
                        selected_hands,
                        current_board,
                        player_count,
                        matchup_weight,
                        pot_shares,
                        total_weight,
                        rank_table=rank_table,
                        binomial_table=binomial_table,
                    )

                    # Advance to the next board combination.
                    j = missing_board - 1

                    while (
                        j >= 0
                        and runout_indices[j]
                        == available_count - missing_board + j
                    ):
                        j -= 1

                    if j < 0:
                        break

                    runout_indices[j] += 1

                    for k in range(j + 1, missing_board):
                        runout_indices[k] = (
                            runout_indices[k - 1] + 1
                        )

        # Advance to the next combination of player hands.
        p = player_count - 1

        while p >= 0:
            hand_indices[p] += 1

            if hand_indices[p] < len(hands[p]):
                break

            hand_indices[p] = 0
            p -= 1

        if p < 0:
            break

    return pot_shares, total_weight



@njit(parallel=True, cache=True)
def _run_exact_equity_kernel_parallel(
    hands,
    masks,
    weights,
    board,
    board_mask,
    mode,
    requested_workers,
    rank_table,
    binomial_table
):
    """Enumerate exact equity in balanced, thread-local Cartesian partitions."""
    player_count = len(hands)
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len

    total_matchups = 1
    for player in range(player_count):
        total_matchups *= len(hands[player])

    worker_count = min(requested_workers, total_matchups)
    # Pad final results so independent workers never update adjacent cache lines.
    worker_results = np.zeros((worker_count, 32), dtype=np.float64)

    for worker in prange(worker_count):
        start = total_matchups * worker // worker_count
        end = total_matchups * (worker + 1) // worker_count
        worker_shares = np.zeros(player_count, dtype=np.float64)
        worker_total = 0.0
        selected_hands = np.empty((player_count, mode), dtype=np.uint8)
        current_board = np.empty(5, dtype=np.uint8)
        available = np.empty(52, dtype=np.uint8)
        runout_indices = np.empty(5, dtype=np.int64)
        hand_indices = np.empty(player_count, dtype=np.int64)

        if board is not None:
            for card in range(board_len):
                current_board[card] = board[card]

        for matchup in range(start, end):
            remainder = matchup
            for player in range(player_count - 1, -1, -1):
                hand_count = len(hands[player])
                hand_indices[player] = remainder % hand_count
                remainder //= hand_count

            used_mask = board_mask
            matchup_weight = 1.0
            valid = True

            for player in range(player_count):
                hand_index = hand_indices[player]
                hand_mask = masks[player][hand_index]

                if used_mask & hand_mask:
                    valid = False
                    break

                used_mask |= hand_mask
                matchup_weight *= weights[player][hand_index]

                if matchup_weight == 0.0:
                    valid = False
                    break

                for card in range(mode):
                    selected_hands[player, card] = hands[player][hand_index, card]

            if valid:
                if missing_board == 0:
                    worker_total = _accumulate_exact_showdown(
                        selected_hands,
                        current_board,
                        player_count,
                        matchup_weight,
                        worker_shares,
                        worker_total,
                        rank_table=rank_table,
                        binomial_table=binomial_table,
                    )
                else:
                    available_count = 0
                    for card in range(52):
                        bit = np.uint64(1) << np.uint64(card)
                        if (used_mask & bit) == 0:
                            available[available_count] = card
                            available_count += 1

                    for card in range(missing_board):
                        runout_indices[card] = card

                    while True:
                        for card in range(missing_board):
                            current_board[board_len + card] = available[
                                runout_indices[card]
                            ]

                        worker_total = _accumulate_exact_showdown(
                            selected_hands,
                            current_board,
                            player_count,
                            matchup_weight,
                            worker_shares,
                            worker_total,
                            rank_table=rank_table,
                            binomial_table=binomial_table,
                        )

                        index = missing_board - 1
                        while (
                            index >= 0
                            and runout_indices[index]
                            == available_count - missing_board + index
                        ):
                            index -= 1

                        if index < 0:
                            break

                        runout_indices[index] += 1
                        for card in range(index + 1, missing_board):
                            runout_indices[card] = runout_indices[card - 1] + 1

        for player in range(player_count):
            worker_results[worker, player] = worker_shares[player]
        worker_results[worker, 8] = worker_total

    # Sum worker-local accumulators in a stable order to avoid write races.
    pot_shares = np.zeros(player_count, dtype=np.float64)
    total_weight = 0.0
    for worker in range(worker_count):
        total_weight += worker_results[worker, 8]
        for player in range(player_count):
            pot_shares[player] += worker_results[worker, player]

    return pot_shares, total_weight



def _estimate_exact_showdowns(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None,
) -> tuple[int, int]:
    """Return candidate hand matchups and the conservative showdown estimate."""
    player_count = len(players)
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len
    remaining_cards = 52 - board_len - player_count * mode
    candidate_matchups = prod(len(player.hands) for player in players)
    estimated_showdowns = candidate_matchups * comb(remaining_cards, missing_board)
    return candidate_matchups, estimated_showdowns



def run_exact_equity(
    players: list[EncodedRange],
    mode: Mode,
    rank_lookup: RankLookup,
    board: NDArray[np.uint8] | None = None,
    max_showdowns: int | None = _EXACT_MAX_SHOWDOWNS,
) -> list[float]:
    """Calculate exact weighted equity with the suitable Numba kernel.

    Small estimated workloads stay sequential; larger workloads partition the
    Cartesian product across the configured Numba threads.

    Args:
        players: Validated, board-filtered player ranges with strictly
            positive weights.
        mode: Number of hole cards per player (4 or 5).
        board: Known community cards, or None for preflop.
        max_showdowns: Maximum estimated showdown evaluations.
            None disables the limit.

    Returns:
        Exact equity for each player as a fraction from 0 to 1.

    Raises:
        ValueError: If no compatible positive-weight matchups exist.
        RuntimeError: If the estimated workload exceeds max_showdowns.
    """
    player_count = len(players)

    board_mask = (
        np.uint64(0)
        if board is None
        else build_mask(board)
    )

    # Conservative upper bound that ignores collisions.
    candidate_matchups, estimated_showdowns = _estimate_exact_showdowns(
        players, mode, board
    )

    if (
        max_showdowns is not None
        and estimated_showdowns > max_showdowns
    ):
        raise RuntimeError(
            "Exact calculation exceeds the workload limit: "
            f"{estimated_showdowns:,} estimated showdowns"
        )

    # Preserve the validated hands and masks directly; only weights need
    # conversion and normalization for numerically stable aggregation.
    hands = List()
    masks = List()
    weights = List()

    for player in players:
        player_weights = player.weights.astype(np.float64)
        player_weights /= player_weights.sum(dtype=np.float64)

        hands.append(player.hands)
        masks.append(player.masks)
        weights.append(player_weights)

    # Benchmarks show a parallel benefit from about 1,000 estimated showdowns.
    # Keep one-matchup workloads sequential because prange cannot split them.
    if (
        estimated_showdowns > _EXACT_PARALLEL_MIN_SHOWDOWNS
        and candidate_matchups > 1
        and get_num_threads() > 1
    ):
        pot_shares, total_weight = _run_exact_equity_kernel_parallel(
            hands,
            masks,
            weights,
            board,
            board_mask,
            mode,
            get_num_threads(),
            rank_table=rank_lookup.rank_table,
            binomial_table=rank_lookup.binomial_table,
        )
    else:
        pot_shares, total_weight = _run_exact_equity_kernel(
            hands=hands,
            masks=masks,
            weights=weights,
            board=board,
            board_mask=board_mask,
            mode=mode,
            rank_table=rank_lookup.rank_table,
            binomial_table=rank_lookup.binomial_table,
        )

    if total_weight == 0:
        raise ValueError(
            "No compatible positive-weight hand combinations exist"
        )

    return [
        float(
            pot_shares[i] / total_weight
        )
        for i in range(player_count)
    ]
