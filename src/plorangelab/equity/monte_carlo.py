"""Adaptive Monte Carlo equity engine with Numba sampling kernels."""

from collections.abc import Callable
from math import sqrt
from threading import Event

import numpy as np
from numba import njit, prange
from numba.typed import List
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange, build_mask
from plorangelab.rank_combination import RankLookup
from plorangelab.types import Mode

from .evaluator import numba_evaluate_hands
from .exceptions import SamplingExhaustedError
from .results import MonteCarloResult, StopReason

_MAX_SIMULATIONS = 1_200_000
_MAX_BATCH_SIZE = 100_000  # About 8 MiB for six-player PLO5 batch buffers.
_MAX_BATCH_ATTEMPTS = 100_000  # Maximum work between coordinator checks.


@njit(cache=True)
def check_collisions(masks: NDArray[np.uint64]) -> bool:
    """Return whether any two selected hand masks contain a shared card.

    Args:
        masks: One 52-card bitmask per selected player hand.

    Returns:
        True if at least two hands overlap; otherwise False.
    """
    used_mask = np.uint64(0)
    for mask in masks:
        if used_mask & mask:
            return True
        used_mask |= mask

    return False


@njit(cache=True)
def deal_card(deck: np.uint64, rng: np.random.Generator) -> tuple[int, np.uint64]:
    """Draw one uniformly random available card and return the updated deck mask.

    Args:
        deck: Bitmask whose set bits represent cards still in the deck.
        rng: Random generator used for reproducible draws.

    Returns:
        A tuple of the selected card ID (0–51) and the deck mask with that
        card's bit cleared.

    Raises:
        ValueError: If no cards remain in the deck.
    """
    if deck == 0:
        raise ValueError("Deck is empty")
    # Rejection sampling over card IDs produces a uniform draw from
    # the remaining cards, regardless of which deck bits are set.
    while True:
        pos = int(rng.integers(0, 52))
        bit = np.uint64(1) << pos

        if deck & bit:
            deck ^= bit
            return pos, deck


def prepare_probabilities(players: list[EncodedRange]) -> list[NDArray[np.float64]]:
    """Convert each player's hand weights to a probability distribution.

    Weight calculations use float64 for reliable normalization. Every input
    hand has positive weight, so each hand receives positive sampling
    probability.

    Returns:
        One float64 probability array per player, aligned with that player's
        encoded hands and masks.

    Raises:
        ValueError: If a player has no finite positive total weight.
    """
    probabilities = []

    for i, player in enumerate(players):
        weights = player.weights.astype(np.float64)
        total = weights.sum()

        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"Player {i + 1} has no positive-weight hands after board filtering"
            )

        probabilities.append(weights / total)

    return probabilities


@njit(parallel=True, cache=True)
def _run_monte_carlo_batch(
    hands: list[NDArray[np.uint8]],
    masks: list[NDArray[np.uint64]],
    cumulative_probabilities: list[NDArray[np.float64]],
    mode: Mode,
    board: NDArray[np.uint8] | None,
    board_mask: np.uint64,
    batch_size: int,
    attempt_budget: int,
    rng: np.random.Generator,
    rank_table: NDArray[np.uint32],
    binomial_table: NDArray[np.uint32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int]:
    """Sample and evaluate up to ``batch_size`` valid Omaha showdowns.

    Each attempt independently draws one weighted hand per player. Colliding
    hand selections are discarded, and missing board cards are drawn uniformly
    from the remaining deck. Each valid showdown awards an equal pot fraction
    to all players tied for the highest hand rank. The arrays allocated here
    are bounded by the requested batch size, not by the total simulation goal.

    Sampling and board generation are performed sequentially, while
    completed showdowns are evaluated in parallel using Numba's prange.
    The batch may return fewer valid simulations than requested if its
    attempt budget is exhausted.

    Args:
        hands: Board-filtered encoded hands for each player.
        masks: Corresponding hand masks.
        cumulative_probabilities: Weighted sampling CDFs, one per player.
        mode: Number of hole cards per hand (4 or 5).
        board: Known community cards, or None before the flop.
        board_mask: Bitmask of known community cards.
        batch_size: Maximum valid showdowns to sample in this call.
        attempt_budget: Maximum hand-selection attempts allowed in this call.
        rng: Shared generator, retained by the Python coordinator across calls.

    Returns:
        Per-player sums of pot shares, sums of squared pot shares, the number
        of valid showdowns, and the number of sampling attempts.
    """
    player_count = len(hands)
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len

    deck = np.uint64((1 << 52) - 1)
    deck &= ~board_mask

    current_board = np.empty(5, dtype=np.uint8)

    if board is not None:
        current_board[:board_len] = board

    selected_masks = np.empty(player_count, dtype=np.uint64)
    selected_hands = np.empty((player_count, mode), dtype=np.uint8)

    simulations = 0
    attempts = 0
    attempt_budget = min(attempt_budget, _MAX_BATCH_ATTEMPTS)
    sampled_hands = np.empty((batch_size, player_count, mode), dtype=np.uint8)
    sampled_boards = np.empty((batch_size, 5), dtype=np.uint8)

    while simulations < batch_size and attempts < attempt_budget:
        attempts += 1
        deck_state = deck

        # Independently sample a weighted hand for each player.
        for i, player_hands in enumerate(hands):
            choice = np.searchsorted(
                cumulative_probabilities[i], rng.random(), side="right"
            )

            selected_masks[i] = masks[i][choice]
            selected_hands[i] = player_hands[choice]

        # Reject the entire selection if hands overlap.
        if check_collisions(selected_masks):
            continue

        # Remove selected hands from the remaining deck.
        for hand_mask in selected_masks:
            deck_state &= ~hand_mask

        # Deal any missing community cards.
        for j in range(missing_board):
            card, deck_state = deal_card(deck_state, rng)
            current_board[board_len + j] = card

        sampled_hands[simulations] = selected_hands
        sampled_boards[simulations] = current_board

        simulations += 1

    pot_shares = np.zeros((simulations, player_count), dtype=np.float64)

    for sim in prange(simulations):
        scores = numba_evaluate_hands(
            sampled_hands[sim],
            sampled_boards[sim],
            rank_table=rank_table,
            binomial_table=binomial_table,
        )

        highest_score = scores[0]
        winners_count = 1

        for player in range(1, player_count):
            if scores[player] > highest_score:
                highest_score = scores[player]
                winners_count = 1
            elif scores[player] == highest_score:
                winners_count += 1

        share = 1.0 / winners_count

        for player in range(player_count):
            if scores[player] == highest_score:
                pot_shares[sim, player] = share

    share_sums = np.zeros(player_count, dtype=np.float64)
    squared_share_sums = np.zeros(player_count, dtype=np.float64)

    # Aggregate sequentially for deterministic results and race-free writes.
    for sim in range(simulations):
        for player in range(player_count):
            share = pot_shares[sim, player]
            share_sums[player] += share
            squared_share_sums[player] += share * share

    return share_sums, squared_share_sums, simulations, attempts


def _prepare_monte_carlo(
    players: list[EncodedRange],
    board: NDArray[np.uint8] | None,
    seed: int,
) -> tuple[
    List,
    List,
    List,
    np.uint64,
    np.random.Generator,
]:
    """Prepare encoded ranges for Numba-compiled Monte Carlo sampling.

    Normalizes each player's weights, constructs cumulative probability
    distributions for weighted hand selection, and converts the input
    arrays into Numba typed lists.

    Returns:
        A tuple containing the hands, masks, cumulative probabilities,
        known board mask, and a seeded NumPy random generator.
    """
    probabilities = prepare_probabilities(players)
    hands = List([player.hands for player in players])
    masks = List([player.masks for player in players])
    cumulative = List()

    for probability in probabilities:
        cdf = np.cumsum(probability)
        cdf[-1] = 1.0  # Preserve the endpoint safeguard against roundoff.
        cumulative.append(cdf)

    board_mask = np.uint64(0) if board is None else build_mask(board)
    return hands, masks, cumulative, board_mask, np.random.default_rng(seed)


@njit(cache=True)
def _probe_collision_acceptance(
    masks: List,
    cumulative_probabilities: List,
    trials: int,
    rng: np.random.Generator,
) -> int:
    """Measure the acceptance rate of independent weighted hand sampling.

    Samples one hand per player using their cumulative probability
    distributions and counts selections without overlapping cards.

    The player ranges are assumed to have already been filtered against
    known board cards.

    Args:
        masks: Encoded hand masks for each player.
        cumulative_probabilities: Weighted sampling CDFs for each player.
        trials: Number of independent matchup selections to attempt.
        rng: Random generator used for sampling.

    Returns:
        Number of collision-free selections among the attempted trials.
    """
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


def _equity_statistics(
    share_sums: NDArray[np.float64],
    squared_share_sums: NDArray[np.float64],
    simulations: int,
    confidence_multiplier: float,
) -> tuple[list[float], list[float | None]]:
    """Calculate Monte Carlo equity estimates and approximate margins of error.

    Each showdown contributes a pot share between 0 and 1, including
    fractional shares for ties.

    Equity is the mean pot share. The margin of error is estimated from
    the sample variance of pot shares and the supplied confidence
    multiplier.

    Args:
        share_sums: Sum of observed pot shares for each player.
        squared_share_sums: Sum of squared pot shares for each player.
        simulations: Number of valid showdowns.
        confidence_multiplier: Multiplier applied to the estimated
            standard error.

    Returns:
        A tuple containing per-player equity estimates and margins of
        error. Margins are None when fewer than two samples are available.

    Notes:
        Margins are approximate and do not account for repeated testing
        or adaptive stopping.
    """
    equities: list[float] = []
    margins: list[float | None] = []

    for i, (share_sum, squared_sum) in enumerate(zip(share_sums, squared_share_sums)):
        equity = float(share_sum / simulations)
        if simulations < 2:
            margin = None
        else:
            # Pot shares can be fractional due to ties, so estimate variance
            # from the observed shares rather than assuming Bernoulli outcomes.
            variance = max(
                float(
                    (squared_sum - share_sum * share_sum / simulations)
                    / (simulations - 1)
                ),
                0.0,
            )
            margin = confidence_multiplier * sqrt(variance / simulations)
        equities.append(equity)
        margins.append(margin)

    return equities, margins


def _run_batched_monte_carlo(
    hands: List,
    masks: List,
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
    """Run adaptive rejection-sampling Monte Carlo in bounded batches.

    Coordinates Numba sampling kernels while accumulating completed
    showdowns across batches.

    Stops when all player margins of error meet the target after the
    minimum sample count, the maximum simulation count is reached,
    the attempt limit is exhausted, or cancellation is requested.

    Cancellation is checked between batches. Intermediate progress
    callbacks are invoked after completed nonterminal batches.

    Args:
        hands: Numba typed list of encoded player hands.
        masks: Corresponding hand masks.
        cumulative_probabilities: Weighted sampling CDFs for each player.
        mode: Number of hole cards per player (4 or 5).
        board: Known community cards, or None for preflop.
        board_mask: Bitmask of known community cards.
        rng: Random generator shared across batches.
        requested_samples: Maximum number of valid showdowns.
        batch_size: Maximum number of valid showdowns per batch.
        attempt_limit: Global maximum number of sampling attempts,
            including rejected selections.
        target_moe: Target absolute margin of error for every player.
        min_samples: Minimum valid showdowns required for precision-based
            stopping.
        confidence_multiplier: Multiplier applied to estimated standard
            errors.
        cancel_event: Optional cancellation event.
        on_progress: Optional callback receiving intermediate results.
        rank_lookup: Precomputed five-card hand ranking tables.

    Returns:
        MonteCarloResult containing per-player equity estimates, margins
        of error, simulation and attempt counts, and the stopping reason.

    Raises:
        ValueError: If simulation or batch limits exceed the configured
            maximum values.
        InterruptedError: If cancellation occurs before any valid
            showdowns have been completed.
        SamplingExhaustedError: If no valid showdowns are produced within
            the attempt limit.
    """
    if requested_samples > _MAX_SIMULATIONS:
        raise ValueError(f"requested_samples must not exceed {_MAX_SIMULATIONS:,}")
    if batch_size > _MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must not exceed {_MAX_BATCH_SIZE:,}")

    player_count = len(hands)
    share_sums = np.zeros(player_count, dtype=np.float64)
    squared_share_sums = np.zeros(player_count, dtype=np.float64)
    simulations = 0
    attempts = 0

    while simulations < requested_samples:
        # Cancellation is observed only at batch boundaries. If nothing has
        # completed yet, there is no estimate to return.
        if cancel_event is not None and cancel_event.is_set():
            if simulations == 0:
                raise InterruptedError(
                    "Monte Carlo calculation cancelled before any simulations completed"
                )
            stop_reason: StopReason = "cancelled"
            break

        attempts_remaining = attempt_limit - attempts
        if attempts_remaining <= 0:
            stop_reason = "attempt_limit"
            break

        batch_count = min(batch_size, requested_samples - simulations)

        # Stop the batch at the minimum sample threshold so the first
        # precision check can occur without overshooting that threshold.
        if simulations < min_samples:
            batch_count = min(batch_count, min_samples - simulations)

        batch_sums, batch_squared_sums, batch_simulations, batch_attempts = (
            _run_monte_carlo_batch(
                hands=hands,
                masks=masks,
                cumulative_probabilities=cumulative_probabilities,
                mode=mode,
                board=board,
                board_mask=board_mask,
                batch_size=batch_count,
                attempt_budget=min(attempts_remaining, _MAX_BATCH_ATTEMPTS),
                rng=rng,
                rank_table=rank_lookup.rank_table,
                binomial_table=rank_lookup.binomial_table,
            )
        )

        # The batch kernel evaluates all valid hands before returning, even if
        # it exhausted its attempt allowance partway through the batch.
        share_sums += batch_sums
        squared_share_sums += batch_squared_sums
        simulations += batch_simulations
        attempts += batch_attempts

        if simulations > 0 and (on_progress is not None or simulations >= min_samples):
            equities, margins = _equity_statistics(
                share_sums,
                squared_share_sums,
                simulations,
                confidence_multiplier,
            )

        if simulations >= min_samples and all(
            margin is not None and margin <= target_moe for margin in margins
        ):
            stop_reason = "target_reached"
            break

        if simulations >= requested_samples:
            stop_reason = "max_simulations"
            break

        if attempts >= attempt_limit:
            stop_reason = "attempt_limit"
            break

        if batch_attempts == 0:
            # Defensive guard against a faulty or substituted batch kernel.
            stop_reason = "attempt_limit"
            break

        if on_progress is not None and batch_simulations > 0:
            on_progress(
                MonteCarloResult(
                    equities=equities,
                    margins_of_error=margins,
                    simulations=simulations,
                    attempts=attempts,
                    stop_reason=None,
                    sampler="rejection",
                )
            )

    else:
        stop_reason = "max_simulations"

    if simulations == 0:
        raise SamplingExhaustedError(
            f"Sampling limit exceeded: 0/{requested_samples} valid simulations "
            f"after {attempts} attempts"
        )

    equities, margins = _equity_statistics(
        share_sums,
        squared_share_sums,
        simulations,
        confidence_multiplier,
    )
    return MonteCarloResult(
        equities=equities,
        margins_of_error=margins,
        simulations=simulations,
        attempts=attempts,
        stop_reason=stop_reason,
    )
