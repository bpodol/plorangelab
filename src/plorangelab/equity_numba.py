from dataclasses import dataclass
from itertools import combinations
from math import comb, prod, sqrt
from threading import Event
from typing import Literal

import numpy as np
from numba import njit, prange
from numba.typed import List
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange, Mode, build_mask
from plorangelab.evaluator_numba import numba_evaluate_hands

_MAX_SIMULATIONS = 1_000_000
_MAX_BATCH_SIZE = 100_000  # About 8 MiB for six-player PLO5 batch buffers.
_MAX_BATCH_ATTEMPTS = 100_000  # Maximum work between coordinator checks.


StopReason = Literal[
    "target_reached", "max_simulations", "cancelled", "attempt_limit"
]


@dataclass(frozen=True)
class MonteCarloResult:
    """Equity estimates and stopping diagnostics for an adaptive run."""

    equities: dict[str, float]
    margins_of_error: dict[str, float]
    simulations: int
    attempts: int
    stop_reason: StopReason

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


def validate_equity_inputs(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None,
    samples: int,
    seed: int,
    method: Literal["exact", "monte-carlo"]
) -> None:
    """Validate the public equity API's parameters and encoded range invariants.

    Checks player count, PLO mode, board cards, sample count, seed, array
    shapes and dtypes, card uniqueness, hand-mask consistency, and strictly
    positive finite weights.
    Does not check whether a compatible multi-player matchup exists.

    Raises:
        ValueError: If any parameter, board, or encoded range is invalid.
    """
    if mode not in (4, 5):
        raise ValueError("Mode must be 4 or 5")

    if not 2 <= len(players) <= 6:
        raise ValueError("Expected 2 to 6 players")

    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples <= 0:
        raise ValueError("Samples must be a positive integer")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    if board is not None:
        if (
            not isinstance(board, np.ndarray)
            or board.dtype != np.uint8
            or board.ndim != 1
        ):
            raise ValueError("Board must be a 1D uint8 array")

        if len(board) not in (0, 3, 4, 5):
            raise ValueError("Board must contain 0, 3, 4, or 5 cards")

        if np.any(board >= 52):
            raise ValueError("Board contains invalid card IDs")

        if len(np.unique(board)) != len(board):
            raise ValueError("Board contains duplicate cards")

    for i, player in enumerate(players):
        if not isinstance(player, EncodedRange):
            raise ValueError(f"Player {i + 1} must be an EncodedRange")

        if (
            not isinstance(player.hands, np.ndarray)
            or player.hands.dtype != np.uint8
            or player.hands.ndim != 2
            or player.hands.shape[1] != mode
        ):
            raise ValueError(f"Player {i + 1} has invalid hands")

        n = len(player.hands)

        if n == 0:
            raise ValueError(f"Player {i + 1} has an empty range")

        if (
            not isinstance(player.masks, np.ndarray)
            or player.masks.dtype != np.uint64
            or player.masks.shape != (n,)
        ):
            raise ValueError(f"Player {i + 1} has invalid masks")

        if (
            not isinstance(player.weights, np.ndarray)
            or player.weights.ndim != 1
            or player.weights.shape != (n,)
            or not np.issubdtype(player.weights.dtype, np.floating)
        ):
            raise ValueError(f"Player {i + 1} has invalid weights")

        if np.any(player.hands >= 52):
            raise ValueError(f"Player {i + 1} contains invalid cards")

        sorted_hands = np.sort(player.hands, axis=1)

        if np.any(sorted_hands[:, 1:] == sorted_hands[:, :-1]):
            raise ValueError(f"Player {i + 1} contains duplicate cards within a hand")

        expected_masks = np.zeros(n, dtype=np.uint64)

        for j in range(mode):
            expected_masks |= (
                np.uint64(1) << player.hands[:, j].astype(np.uint64)
            )

        if not np.array_equal(player.masks, expected_masks):
            raise ValueError(f"Player {i + 1} contains incorrect masks")

        if len(np.unique(player.masks)) != n:
            raise ValueError(f"Player {i + 1} contains duplicate hands")

        weights = player.weights.astype(np.float64)
        total = weights.sum()

        if (
            not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
            or not np.isfinite(total)
            or total <= 0
        ):
            raise ValueError(f"Player {i + 1} has invalid weights")

    if not isinstance(method, str) or method not in ("exact", "monte-carlo"):
        raise ValueError("Method must be 'exact' or 'monte-carlo'")

def sanitize_range(player: EncodedRange, board_mask: np.uint64) -> EncodedRange:
    """Return a new range with all board-blocked hands removed.

    Preserves alignment between hands, masks, and weights; leaves the original
    range unchanged.

    Raises:
        ValueError: If no hands survive board-card filtering.
    """
    valid = (player.masks & board_mask) == 0

    if not valid.any():
        raise ValueError("No combinations left after removing board cards")

    return EncodedRange(
        hands=player.hands[valid],
        masks=player.masks[valid],
        weights=player.weights[valid],
    )


def prepare_ranges(players: list[EncodedRange], board: NDArray[np.uint8] | None) -> list[EncodedRange]:
    """Filter all player ranges against the known community cards.

    Args:
        players: Validated encoded ranges.
        board: Known board cards, or None for a board with no known cards.

    Returns:
        Board-compatible ranges. With no board, returns a shallow copy of
        the players list; otherwise returns filtered EncodedRange instances.
    """
    if board is None or len(board) == 0:
        return players.copy()

    board_mask = build_mask(board)

    return [
        sanitize_range(player, board_mask)
        for player in players
    ]

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


@njit(cache=True)
def _run_exact_equity_kernel(
    hands,
    masks,
    weights,
    board,
    board_mask,
    mode,
):
    """
    Enumerate all compatible matchups and board runouts.

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

    # Reusable buffers.
    available = np.empty(52, dtype=np.uint8)
    runout_indices = np.empty(5, dtype=np.int64)
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

                scores = numba_evaluate_hands(
                    selected_hands,
                    current_board,
                )

                highest_score = scores[0]

                for p in range(1, player_count):
                    if scores[p] > highest_score:
                        highest_score = scores[p]

                winners_count = 0

                for p in range(player_count):
                    if scores[p] == highest_score:
                        winners_count += 1

                share = matchup_weight / winners_count

                for p in range(player_count):
                    if scores[p] == highest_score:
                        pot_shares[p] += share

                total_weight += matchup_weight

                # A complete river board has one runout.
                if missing_board == 0:
                    break

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

def run_exact_equity(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None = None,
    max_showdowns: int | None = 100_000,
) -> dict[str, float]:
    """Calculate exact weighted equity using a sequential Numba kernel.

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
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len

    board_mask = (
        np.uint64(0)
        if board is None
        else build_mask(board)
    )

    # Conservative upper bound that ignores collisions.
    remaining_cards = 52 - board_len - player_count * mode

    estimated_showdowns = (
        prod(len(player.hands) for player in players)
        * comb(remaining_cards, missing_board)
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

    # Enumerate and evaluate entirely inside Numba.
    pot_shares, total_weight = _run_exact_equity_kernel(
        hands=hands,
        masks=masks,
        weights=weights,
        board=board,
        board_mask=board_mask,
        mode=mode,
    )

    if total_weight == 0:
        raise ValueError(
            "No compatible positive-weight hand combinations exist"
        )

    return {
        f"player {i + 1}:": float(
            pot_shares[i] / total_weight
        )
        for i in range(player_count)
    }


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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
    """Prepare validated Numba inputs and one RNG shared by all batches."""
    valid_players = prepare_ranges(players, board)
    probabilities = prepare_probabilities(valid_players)
    hands = List([player.hands for player in valid_players])
    masks = List([player.masks for player in valid_players])
    cumulative = List()

    for probability in probabilities:
        cdf = np.cumsum(probability)
        cdf[-1] = 1.0  # Preserve the endpoint safeguard against roundoff.
        cumulative.append(cdf)

    board_mask = np.uint64(0) if board is None else build_mask(board)
    return hands, masks, cumulative, board_mask, np.random.default_rng(seed)


def _equity_statistics(
    share_sums: NDArray[np.float64],
    squared_share_sums: NDArray[np.float64],
    simulations: int,
    confidence_multiplier: float,
) -> tuple[dict[str, float], dict[str, float]]:
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
    adaptive: bool,
    target_moe: float,
    min_samples: int,
    confidence_multiplier: float,
    cancel_event: Event | None = None,
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
        if adaptive and simulations < min_samples:
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

        if adaptive and simulations >= min_samples:
            _, margins = _equity_statistics(
                share_sums,
                squared_share_sums,
                simulations,
                confidence_multiplier,
            )
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


def compute_equity_adaptive(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None = None,
    *,
    target_moe: float = 0.001,
    confidence_multiplier: float = 1.96,
    min_samples: int = 10_000,
    max_samples: int = 1_000_000,
    batch_size: int = 10_000,
    seed: int = 3,
    cancel_event: Event | None = None,
) -> MonteCarloResult:
    """Estimate equity in bounded batches until the estimated precision target.

    The margins of error are approximate plug-in estimates. Repeatedly checking
    them and stopping conditionally (optional stopping) means they are not
    formal confidence bounds and do not guarantee 95% coverage. Rare outcomes
    that have not yet appeared in the sample can also make the estimate
    misleading.

    If cancellation is set before the first batch, this raises
    :class:`InterruptedError` because no estimate exists. Cancellation between
    batches returns completed work with ``stop_reason="cancelled"``.
    """
    if cancel_event is not None:
        if not isinstance(cancel_event, Event):
            raise ValueError("cancel_event must be a threading.Event or None")
        if cancel_event.is_set():
            raise InterruptedError(
                "Monte Carlo calculation cancelled before any simulations "
                "(before validation or preparation)"
            )

    max_samples = _positive_integer(max_samples, "max_samples")
    min_samples = _positive_integer(min_samples, "min_samples")
    batch_size = _positive_integer(batch_size, "batch_size")
    if max_samples > _MAX_SIMULATIONS:
        raise ValueError(f"max_samples must not exceed {_MAX_SIMULATIONS:,}")
    if batch_size > _MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must not exceed {_MAX_BATCH_SIZE:,}")
    if min_samples > max_samples:
        raise ValueError("min_samples must not exceed max_samples")
    if (
        isinstance(target_moe, (bool, np.bool_))
        or not isinstance(target_moe, (int, float, np.number))
        or not np.isfinite(target_moe)
        or target_moe < 0
    ):
        raise ValueError("target_moe must be a finite non-negative number")
    if (
        isinstance(confidence_multiplier, (bool, np.bool_))
        or not isinstance(confidence_multiplier, (int, float, np.number))
        or not np.isfinite(confidence_multiplier)
        or confidence_multiplier <= 0
    ):
        raise ValueError("confidence_multiplier must be a finite positive number")
    validate_equity_inputs(
        players, mode, board, max_samples, seed, "monte-carlo"
    )
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError(
            "Monte Carlo calculation cancelled before preparation"
        )
    hands, masks, cumulative, board_mask, rng = _prepare_monte_carlo(
        players, board, seed
    )
    return _run_batched_monte_carlo(
        hands,
        masks,
        cumulative,
        mode,
        board,
        board_mask,
        rng,
        requested_samples=max_samples,
        batch_size=batch_size,
        attempt_limit=max(10_000, max_samples * 20),
        adaptive=True,
        target_moe=float(target_moe),
        min_samples=min_samples,
        confidence_multiplier=float(confidence_multiplier),
        cancel_event=cancel_event,
    )


def compute_equity(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None = None,
    samples: int = 100_000,
    seed: int = 3,
    method: Literal["exact", "monte-carlo"] = "monte-carlo",
    batch_size: int = 10_000,
) -> dict[str, float]:
    """Calculate equity for two to six weighted PLO4/PLO5 ranges.

    Monte Carlo calls retain fixed-sample behavior: ``samples`` means exactly
    that many valid showdowns, processed in batches no larger than
    ``batch_size``. Use :func:`compute_equity_adaptive` for precision-based
    stopping and its diagnostic result. Input ranges are not modified.
    """
    batch_size = _positive_integer(batch_size, "batch_size")
    if method == "monte-carlo":
        samples = _positive_integer(samples, "samples")
        if samples > _MAX_SIMULATIONS:
            raise ValueError(f"samples must not exceed {_MAX_SIMULATIONS:,}")
        if batch_size > _MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must not exceed {_MAX_BATCH_SIZE:,}")

    validate_equity_inputs(players, mode, board, samples, seed, method)

    if method == "exact":
        valid_players = prepare_ranges(players, board)
        return run_exact_equity(
            players=valid_players,
            mode=mode,
            board=board,
        )

    hands, masks, cumulative, board_mask, rng = _prepare_monte_carlo(
        players, board, seed
    )
    result = _run_batched_monte_carlo(
        hands,
        masks,
        cumulative,
        mode,
        board,
        board_mask,
        rng,
        requested_samples=int(samples),
        batch_size=batch_size,
        attempt_limit=max(10_000, int(samples) * 20),
        adaptive=False,
        target_moe=0.0,
        min_samples=int(samples),
        confidence_multiplier=1.96,
    )
    if result.stop_reason == "attempt_limit":
        raise RuntimeError(
            f"Sampling limit exceeded: {result.simulations}/{samples} valid "
            f"simulations after {result.attempts} attempts"
        )
    return result.equities

