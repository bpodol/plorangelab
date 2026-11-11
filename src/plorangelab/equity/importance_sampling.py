"""Sequential importance sampling for multiway PLO equity estimation.

Samples player hands conditionally on previously selected cards to
reduce collision rejections. Importance weights correct for the
modified sampling distribution.

Uses precomputed compatibility indexes for smaller ranges and
Numba-compiled kernels for sampling and showdown evaluation.
"""

from collections.abc import Callable
from math import sqrt
from threading import Event

import numpy as np
from numba import njit, prange
from numba.typed import List
from numpy.typing import NDArray

from plorangelab.rank_combination import RankLookup
from plorangelab.types import Mode

from .evaluator import numba_evaluate_hands
from .exceptions import SamplingExhaustedError
from .monte_carlo import (
    _MAX_BATCH_ATTEMPTS,
    deal_card,
)
from .results import MonteCarloResult, StopReason

_INDEX_MAX_HANDS = 50_000


@njit(cache=True)
def _probe_collision_acceptance(
    masks: List,
    cumulative_probabilities: List,
    trials: int,
    rng: np.random.Generator,
) -> int:
    """Estimate acceptance of independently sampled player hands."""
    accepted = 0

    for _ in range(trials):
        occupied = np.uint64(0)
        legal = True

        for player in range(len(masks)):
            index = np.searchsorted(
                cumulative_probabilities[player],
                rng.random(),
                side="right",
            )

            mask = masks[player][index]

            if occupied & mask:
                legal = False
                break

            occupied |= mask

        if legal:
            accepted += 1

    return accepted


@njit(cache=True)
def _index_one_range(masks, probabilities):
    """Build compatibility lookup tables for a single weighted range.

    Creates a card-membership bitset identifying which hands contain
    each card, and a lookup table containing the total probability mass
    of every possible subset of eight consecutive hands.

    The subset table enables efficient calculation of the probability
    mass remaining after incompatible hands are excluded.

    Args:
        masks: One uint64 card mask per hand.
        probabilities: Normalized hand probabilities aligned with masks.

    Returns:
        A tuple containing:
            - membership: uint64 bitsets of shape (52, ceil(n / 64)).
            - subset_masses: float64 array of shape (ceil(n / 8), 256).
    """
    n = len(masks)

    membership = np.zeros(
        (52, (n + 63) // 64),
        dtype=np.uint64,
    )
    subset_masses = np.zeros(
        ((n + 7) // 8, 256),
        dtype=np.float64,
    )

    # Each card maps to a bitset of hands containing that card.
    for i in range(n):
        for card in range(52):
            if masks[i] & (np.uint64(1) << np.uint64(card)):
                membership[card, i // 64] |= np.uint64(1) << np.uint64(i % 64)

    # Precompute weighted probability mass for all 256 subsets
    # of each consecutive block of eight hands.
    for block in range(len(subset_masses)):
        for pattern in range(1, 256):
            bit = 0
            while ((pattern >> bit) & 1) == 0:
                bit += 1

            hand_index = block * 8 + bit
            subset_masses[block, pattern] = subset_masses[block, pattern ^ (1 << bit)]

            if hand_index < n:
                subset_masses[block, pattern] += probabilities[hand_index]

    return membership, subset_masses


def _build_compatibility_index(
    masks: List,
    probabilities: List,
) -> tuple[List, List, bool]:
    """Prepare compatibility indexes for conditional hand sampling.

    Builds card-membership and subset-mass lookup tables when every
    player range contains at most _INDEX_MAX_HANDS combinations.

    If any range exceeds this threshold, indexing is disabled for
    all players and empty placeholder arrays are returned. The
    sampler then uses direct compatibility scans instead.

    Args:
        masks: Numba typed list of player hand-mask arrays.
        probabilities: Corresponding normalized probability arrays.

    Returns:
        A tuple containing membership tables, subset-mass tables,
        and a boolean indicating whether indexed sampling is enabled.
    """
    indexed = all(len(player_masks) <= _INDEX_MAX_HANDS for player_masks in masks)

    memberships = List()
    subset_masses = List()

    for player in range(len(masks)):
        if indexed:
            card_bits, masses = _index_one_range(
                masks[player],
                probabilities[player],
            )
        else:
            card_bits = np.empty((0, 0), dtype=np.uint64)
            masses = np.empty((0, 0), dtype=np.float64)

        memberships.append(card_bits)
        subset_masses.append(masses)

    return memberships, subset_masses, indexed


@njit(inline="always")
def _select_indexed_hand(
    occupied,
    probabilities,
    membership,
    subset_masses,
    blocked_words,
    legal_blocks,
    rng,
):
    """Sample a compatible hand using precomputed probability lookup tables.

    Combines the membership bitsets of occupied cards to identify
    blocked hands, then calculates the total probability mass of
    all compatible hands.

    Samples a hand proportionally to its original probability,
    conditional on compatibility with the occupied cards.

    Args:
        occupied: Bitmask of cards already used by the board and
            previously selected player hands.
        probabilities: Normalized probabilities for the current range.
        membership: Card-to-hand membership bitsets.
        subset_masses: Precomputed probability masses for eight-hand
            subsets.
        blocked_words: Reusable buffer for blocked-hand bitsets.
        legal_blocks: Reusable buffer for compatible eight-hand subsets.
        rng: Random generator used for weighted selection.

    Returns:
        A tuple of (selected_hand_index, compatible_probability_mass).
        Returns (-1, 0.0) when no compatible hand exists.
    """

    word_count = membership.shape[1]

    for word in range(word_count):
        blocked_words[word] = np.uint64(0)

    # Combine all hand sets blocked by previously selected cards.
    for card in range(52):
        if occupied & (np.uint64(1) << np.uint64(card)):
            for word in range(word_count):
                blocked_words[word] |= membership[card, word]

    total_mass = 0.0

    for block in range(subset_masses.shape[0]):
        legal = np.uint8(
            (~blocked_words[block // 8] >> np.uint64((block % 8) * 8)) & np.uint64(255)
        )

        legal_blocks[block] = legal
        total_mass += subset_masses[block, int(legal)]

    if total_mass <= 0.0:
        return -1, 0.0

    # Weighted selection using the exact compatible mass.
    target = rng.random() * total_mass
    last_legal = -1

    for block in range(subset_masses.shape[0]):
        allowed = int(legal_blocks[block])
        mass = subset_masses[block, allowed]

        if mass <= 0.0:
            continue

        if target >= mass:
            target -= mass
            continue

        for bit in range(8):
            if allowed & (1 << bit):
                hand_index = block * 8 + bit
                last_legal = hand_index

                if target < probabilities[hand_index]:
                    return hand_index, total_mass

                target -= probabilities[hand_index]

    # Guard against floating-point rounding at the endpoint.
    if last_legal < 0:
        for block in range(subset_masses.shape[0] - 1, -1, -1):
            allowed = int(legal_blocks[block])

            for bit in range(7, -1, -1):
                if allowed & (1 << bit):
                    return block * 8 + bit, total_mass

    return last_legal, total_mass


@njit(parallel=True, cache=True)
def _run_importance_batch(
    hands: List,
    masks: List,
    probabilities: List,
    cumulative_probabilities: List,
    mode: Mode,
    board: NDArray[np.uint8] | None,
    board_mask: np.uint64,
    batch_size: int,
    attempt_budget: int,
    rng: np.random.Generator,
    memberships: List,
    subset_masses: List,
    indexed: bool,
    rank_table: NDArray[np.uint32],
    binomial_table: NDArray[np.uint32],
):
    """Generate and evaluate a batch of importance-weighted Omaha showdowns.

    Samples the first player's hand from its original distribution.
    Subsequent players are sampled conditionally on cards already
    selected, avoiding hand collisions whenever a compatible hand exists.

    Each successful matchup receives an importance weight equal to the
    product of the compatible probability masses for players after
    the first. This corrects for the conditional sampling distribution.

    Missing community cards are sampled uniformly from the remaining
    deck. Sampling is sequential, while completed showdowns are
    evaluated in parallel using Numba's prange.

    The batch may return fewer valid showdowns than requested if the
    attempt budget is exhausted.

    Returns:
        A tuple containing:
            - Sum of importance weights.
            - Sum of squared importance weights.
            - Per-player sums of weighted pot shares.
            - Per-player sums of squared weights times pot shares.
            - Per-player sums of squared weights times squared pot shares.
            - Number of completed showdowns.
            - Number of sampling attempts.
    """
    player_count = len(hands)
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len

    current_board = np.empty(5, dtype=np.uint8)

    if board is not None:
        current_board[:board_len] = board

    deck = np.uint64((1 << 52) - 1) & ~board_mask

    selected_hands = np.empty((player_count, mode), dtype=np.uint8)

    sampled_hands = np.empty(
        (batch_size, player_count, mode),
        dtype=np.uint8,
    )
    sampled_boards = np.empty((batch_size, 5), dtype=np.uint8)
    sampled_weights = np.empty(batch_size, dtype=np.float64)
    max_words = 1
    max_blocks = 1

    if indexed:
        for player in range(player_count):
            max_words = max(
                max_words,
                memberships[player].shape[1],
            )
            max_blocks = max(
                max_blocks,
                subset_masses[player].shape[0],
            )

    blocked_words = np.empty(max_words, dtype=np.uint64)
    legal_blocks = np.empty(max_blocks, dtype=np.uint8)
    simulations = 0
    attempts = 0

    while simulations < batch_size and attempts < min(
        attempt_budget, _MAX_BATCH_ATTEMPTS
    ):
        attempts += 1

        occupied = board_mask
        sample_weight = 1.0
        compatible = True

        for player in range(player_count):
            if player == 0:
                selected_index = np.searchsorted(
                    cumulative_probabilities[player],
                    rng.random(),
                    side="right",
                )

            else:
                if indexed:
                    selected_index, remaining_mass = _select_indexed_hand(
                        occupied,
                        probabilities[player],
                        memberships[player],
                        subset_masses[player],
                        blocked_words,
                        legal_blocks,
                        rng,
                    )
                else:
                    # Retain the original two-pass sampler for oversized ranges.
                    remaining_mass = 0.0

                    for hand_index in range(len(masks[player])):
                        if (occupied & masks[player][hand_index]) == 0:
                            remaining_mass += probabilities[player][hand_index]

                    selected_index = -1

                    if remaining_mass > 0.0:
                        target = rng.random() * remaining_mass
                        running = 0.0
                        last_compatible = -1

                        for hand_index in range(len(masks[player])):
                            if (occupied & masks[player][hand_index]) == 0:
                                last_compatible = hand_index
                                running += probabilities[player][hand_index]

                                if running > target:
                                    selected_index = hand_index
                                    break

                        if selected_index < 0:
                            selected_index = last_compatible

                if remaining_mass <= 0.0:
                    compatible = False
                    break
                # Conditional sampling divides by the compatible probability mass.
                # Multiplying these masses corrects the resulting sampling bias.
                sample_weight *= remaining_mass

            selected_hands[player] = hands[player][selected_index]
            occupied |= masks[player][selected_index]

        if not compatible:
            continue

        current_deck = deck & ~occupied

        for j in range(missing_board):
            card, current_deck = deal_card(current_deck, rng)
            current_board[board_len + j] = card

        sampled_hands[simulations] = selected_hands
        sampled_boards[simulations] = current_board
        sampled_weights[simulations] = sample_weight

        simulations += 1

    shares = np.zeros(
        (simulations, player_count),
        dtype=np.float64,
    )
    # Sampling is sequential because each player's selection depends
    # on previously selected cards. Completed showdowns are independent
    # and can therefore be evaluated in parallel.
    for sim in prange(simulations):
        scores = numba_evaluate_hands(
            sampled_hands[sim],
            sampled_boards[sim],
            rank_table=rank_table,
            binomial_table=binomial_table,
        )

        best = scores[0]

        for player in range(1, player_count):
            best = max(best, scores[player])

        winner_count = 0

        for player in range(player_count):
            if scores[player] == best:
                winner_count += 1

        for player in range(player_count):
            if scores[player] == best:
                shares[sim, player] = 1.0 / winner_count

    weight_sum = 0.0
    squared_weight_sum = 0.0

    weighted_shares = np.zeros(player_count, dtype=np.float64)
    squared_weight_shares = np.zeros(player_count, dtype=np.float64)
    squared_weight_squared_shares = np.zeros(
        player_count,
        dtype=np.float64,
    )

    for sim in range(simulations):
        w = sampled_weights[sim]
        w2 = w * w

        weight_sum += w
        squared_weight_sum += w2

        for player in range(player_count):
            x = shares[sim, player]

            weighted_shares[player] += w * x
            squared_weight_shares[player] += w2 * x
            squared_weight_squared_shares[player] += w2 * x * x

    return (
        weight_sum,
        squared_weight_sum,
        weighted_shares,
        squared_weight_shares,
        squared_weight_squared_shares,
        simulations,
        attempts,
    )


def _importance_statistics(
    weight_sum: float,
    squared_weight_sum: float,
    weighted_shares: NDArray[np.float64],
    squared_weight_shares: NDArray[np.float64],
    squared_weight_squared_shares: NDArray[np.float64],
    confidence_multiplier: float,
) -> tuple[list[float], list[float | None], float]:
    """Calculate self-normalized importance sampling statistics.

    Estimates equity from weighted pot shares and calculates an
    approximate margin of error using squared importance weights
    and residuals around the estimated equity.

    Effective sample size measures the concentration of importance
    weights and may be substantially smaller than the number of
    completed showdowns.

    Args:
        weight_sum: Sum of importance weights.
        squared_weight_sum: Sum of squared importance weights.
        weighted_shares: Per-player sums of weight times pot share.
        squared_weight_shares: Per-player sums of squared weight
            times pot share.
        squared_weight_squared_shares: Per-player sums of squared
            weight times squared pot share.
        confidence_multiplier: Multiplier applied to the estimated
            standard error.

    Returns:
        Per-player equity estimates, approximate margins of error,
        and effective sample size. Margins are None when effective
        sample size is below two.

    Notes:
        The margins are approximate and do not provide formal
        confidence guarantees under adaptive stopping.
    """
    # Kish effective sample size: unequal importance weights reduce
    # the effective amount of information in the sample.
    effective_samples = weight_sum * weight_sum / squared_weight_sum

    equities = []
    margins = []

    for index in range(len(weighted_shares)):
        equity = float(weighted_shares[index] / weight_sum)

        if effective_samples < 2:
            margin = None
        else:
            residual_square = (
                squared_weight_squared_shares[index]
                - 2 * equity * squared_weight_shares[index]
                + equity * equity * squared_weight_sum
            )

            margin = (
                confidence_multiplier * sqrt(max(0.0, residual_square)) / weight_sum
            )

        equities.append(equity)
        margins.append(margin)

    return equities, margins, effective_samples


def _run_importance_monte_carlo(
    hands: List,
    masks: List,
    probabilities: List,
    cumulative_probabilities: List,
    mode: Mode,
    board: NDArray[np.uint8] | None,
    board_mask: np.uint64,
    rng: np.random.Generator,
    *,
    requested_samples: int,
    batch_size: int,
    attempt_limit: int,
    target_moe: float,
    min_samples: int,
    confidence_multiplier: float,
    cancel_event: Event | None = None,
    on_progress: Callable[[MonteCarloResult], None] | None = None,
    rank_lookup: RankLookup,
) -> MonteCarloResult:
    """Coordinate adaptive importance sampling across bounded batches.

    Builds compatibility lookup tables once per calculation, executes
    Numba sampling batches, and accumulates weighted showdown statistics.

    Stops when the target margin of error is achieved for every player
    after reaching the minimum effective sample size, the maximum
    simulation count is reached, the attempt budget is exhausted,
    or cancellation is requested.

    Cancellation is checked between batches. Progress callbacks
    receive intermediate MonteCarloResult instances.

    Args:
        hands: Numba typed list of encoded player hands.
        masks: Corresponding hand-mask arrays.
        probabilities: Normalized hand probabilities for each player.
        cumulative_probabilities: Weighted sampling CDFs.
        mode: Number of hole cards per player (4 or 5).
        board: Known community cards, or None for preflop.
        board_mask: Bitmask of known community cards.
        rng: Random generator shared across batches.
        requested_samples: Maximum number of completed showdowns.
        batch_size: Maximum number of valid showdowns per batch.
        attempt_limit: Global maximum number of sampling attempts.
        target_moe: Target absolute margin of error for every player.
        min_samples: Minimum effective sample size required for
            precision-based stopping.
        confidence_multiplier: Multiplier applied to estimated
            standard errors.
        cancel_event: Optional cancellation event.
        on_progress: Optional callback for intermediate results.
        rank_lookup: Precomputed five-card ranking tables.

    Returns:
        MonteCarloResult containing weighted equity estimates,
        approximate margins of error, simulation and attempt counts,
        stopping reason, and effective sample size.

    Raises:
        InterruptedError: If cancelled before any showdowns are completed.
        SamplingExhaustedError: If no valid showdowns are produced within
            the sampling attempt limit.
    """
    player_count = len(hands)

    weight_sum = 0.0
    squared_weight_sum = 0.0

    weighted_shares = np.zeros(player_count, dtype=np.float64)
    squared_weight_shares = np.zeros(player_count, dtype=np.float64)
    squared_weight_squared_shares = np.zeros(
        player_count,
        dtype=np.float64,
    )

    simulations = 0
    attempts = 0
    memberships, subset_masses, indexed = _build_compatibility_index(
        masks,
        probabilities,
    )
    while simulations < requested_samples:
        if cancel_event is not None and cancel_event.is_set():
            if simulations == 0:
                raise InterruptedError(
                    "Monte Carlo calculation cancelled before any simulations completed"
                )

            stop_reason: StopReason = "cancelled"
            break

        remaining_attempts = attempt_limit - attempts

        if remaining_attempts <= 0:
            stop_reason = "attempt_limit"
            break

        batch_result = _run_importance_batch(
            hands,
            masks,
            probabilities,
            cumulative_probabilities,
            mode,
            board,
            board_mask,
            min(batch_size, requested_samples - simulations),
            min(remaining_attempts, _MAX_BATCH_ATTEMPTS),
            rng,
            memberships,
            subset_masses,
            indexed,
            rank_table=rank_lookup.rank_table,
            binomial_table=rank_lookup.binomial_table,
        )

        (
            batch_weight,
            batch_weight_squared,
            batch_shares,
            batch_squared_shares,
            batch_squared_squared_shares,
            batch_simulations,
            batch_attempts,
        ) = batch_result

        weight_sum += batch_weight
        squared_weight_sum += batch_weight_squared
        weighted_shares += batch_shares
        squared_weight_shares += batch_squared_shares
        squared_weight_squared_shares += batch_squared_squared_shares

        simulations += batch_simulations
        attempts += batch_attempts

        if simulations > 0:
            equities, margins, effective_samples = _importance_statistics(
                weight_sum,
                squared_weight_sum,
                weighted_shares,
                squared_weight_shares,
                squared_weight_squared_shares,
                confidence_multiplier,
            )

            if effective_samples >= min_samples and all(
                margin is not None and margin <= target_moe for margin in margins
            ):
                stop_reason = "target_reached"
                break

            if on_progress is not None and batch_simulations > 0:
                on_progress(
                    MonteCarloResult(
                        equities=equities,
                        margins_of_error=margins,
                        simulations=simulations,
                        attempts=attempts,
                        stop_reason=None,
                        sampler="importance",
                        effective_samples=effective_samples,
                    )
                )

        if simulations >= requested_samples:
            stop_reason = "max_simulations"
            break

        if attempts >= attempt_limit or batch_attempts == 0:
            stop_reason = "attempt_limit"
            break

    else:
        stop_reason = "max_simulations"

    if simulations == 0:
        raise SamplingExhaustedError(
            f"Sampling limit exceeded: 0/{requested_samples} valid simulations "
            f"after {attempts} attempts"
        )

    equities, margins, effective_samples = _importance_statistics(
        weight_sum,
        squared_weight_sum,
        weighted_shares,
        squared_weight_shares,
        squared_weight_squared_shares,
        confidence_multiplier,
    )

    return MonteCarloResult(
        equities=equities,
        margins_of_error=margins,
        simulations=simulations,
        attempts=attempts,
        stop_reason=stop_reason,
        sampler="importance",
        effective_samples=effective_samples,
    )
