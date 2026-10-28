"""Adaptive Monte Carlo equity engine with Numba sampling kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from threading import Event
from typing import Literal

import numpy as np
from numba import njit, prange
from numba.typed import List
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange, Mode, build_mask

from .evaluator import numba_evaluate_hands

_MAX_SIMULATIONS = 1_200_000
_MAX_BATCH_SIZE = 100_000  # About 8 MiB for six-player PLO5 batch buffers.
_MAX_BATCH_ATTEMPTS = 100_000  # Maximum work between coordinator checks.

StopReason = Literal[
    "target_reached", "max_simulations", "cancelled", "attempt_limit"
]


@dataclass(frozen=True)
class MonteCarloResult:
    """Equity estimates and stopping diagnostics for an adaptive run."""

    equities: dict[str, float]
    margins_of_error: dict[str, float | None]
    simulations: int
    attempts: int
    stop_reason: StopReason | None
    sampler: Literal["rejection", "importance"] = "rejection"
    effective_samples: float | None = None


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
                f"Player {i + 1} has no positive-weight hands "
                "after board filtering"
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int]:
    """Sample and evaluate up to ``batch_size`` valid Omaha showdowns.

    Each attempt independently draws one weighted hand per player. Colliding
    hand selections are discarded, and missing board cards are drawn uniformly
    from the remaining deck. Each valid showdown awards an equal pot fraction
    to all players tied for the highest hand rank. The arrays allocated here
    are bounded by the requested batch size, not by the total simulation goal.

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
            choice = np.searchsorted(cumulative_probabilities[i], rng.random(), side="right")

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
    """Prepare validated, board-filtered Numba inputs and a shared RNG."""
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



def _equity_statistics(
    share_sums: NDArray[np.float64],
    squared_share_sums: NDArray[np.float64],
    simulations: int,
    confidence_multiplier: float,
) -> tuple[dict[str, float], dict[str, float | None]]:
    equities: dict[str, float] = {}
    margins: dict[str, float | None] = {}

    for player, (share_sum, squared_sum) in enumerate(
        zip(share_sums, squared_share_sums), start=1
    ):
        label = f"player {player}:"
        equity = float(share_sum / simulations)
        if simulations < 2:
            margin = None
        else:
            variance = max(
                float(
                    (squared_sum - share_sum * share_sum / simulations)
                    / (simulations - 1)
                ),
                0.0,
            )
            margin = confidence_multiplier * sqrt(variance / simulations)
        equities[label] = equity
        margins[label] = margin

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
    on_progress: Callable[[MonteCarloResult], None] | None = None
) -> MonteCarloResult:
    """Coordinate batches and aggregate every completed showdown.

    The global attempt limit is shared across calls to the Numba batch kernel;
    each invocation receives at most ``_MAX_BATCH_ATTEMPTS`` attempts so the
    adaptive caller can observe cancellation between invocations. Any valid
    simulations completed before an invocation exhausts its budget are added
    before the coordinator decides whether to continue or stop.
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
            )
        )

        # The batch kernel evaluates all valid hands before returning, even if
        # it exhausted its attempt allowance partway through the batch.
        share_sums += batch_sums
        squared_share_sums += batch_squared_sums
        simulations += batch_simulations
        attempts += batch_attempts

        if simulations >= 0 and (on_progress is not None or simulations >= min_samples):
            equities, margins = _equity_statistics(
                share_sums,
                squared_share_sums,
                simulations,
                confidence_multiplier,
            )

        if simulations >= min_samples:
            if all(margin is not None and margin <= target_moe for margin in margins.values()):
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
        raise RuntimeError(
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

