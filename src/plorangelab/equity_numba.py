from numpy.typing import NDArray
import numpy as np
from plorangelab.encoder import EncodedRange, build_mask, Mode
from plorangelab.evaluator_numba import numba_evaluate_hands
from itertools import combinations, product
from math import comb, prod
from typing import Literal
from numba import njit, prange
from numba.typed import List

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
    shapes and dtypes, card uniqueness, hand-mask consistency, and weights.
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
            or np.any(weights < 0)
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

    Weight calculations use float64 for reliable normalization. Zero-weight
    hands receive zero sampling probability.

    Returns:
        One float64 probability array per player, aligned with that player's
        encoded hands and masks.

    Raises:
        ValueError: If a player has no positive total weight.
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
def run_monte_carlo(
    hands: list[NDArray[np.uint8]],
    masks: list[NDArray[np.uint64]],
    cumulative_probabilities: list[NDArray[np.float64]],
    mode: Mode,
    board: NDArray[np.uint8] | None,
    board_mask: np.uint64,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Estimate weighted multiway Omaha equity by rejection sampling.

    Each attempt independently draws one weighted hand per player. Colliding
    hand selections are discarded, and missing board cards are drawn uniformly
    from the remaining deck. Each valid showdown awards an equal pot fraction
    to all players tied for the highest hand rank.

    Args:
        players: Validated ranges with board-blocked hands removed.
        probabilities: Per-player sampling probabilities aligned with the ranges.
        mode: Number of hole cards per hand (4 or 5).
        board: Known community cards, or None before the flop.
        samples: Number of successful (collision-free) showdowns to evaluate.
        seed: Seed for reproducible sampling and board completion.

    Returns:
        A player-to-equity mapping with values in [0, 1].

    Raises:
        RuntimeError: If the sampling attempt budget is exhausted before
            collecting the requested number of valid showdowns.
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
    max_attempts = max(10_000, samples * 20)

    sampled_hands = np.empty((samples, player_count, mode), dtype=np.uint8)    

    sampled_boards = np.empty((samples, 5), dtype=np.uint8)   

    while simulations < samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Sampling limit exceeded: {simulations}/{samples} "
                f"valid simulations after {attempts} attempts"
            )

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

    pot_shares = np.zeros((samples, player_count), dtype=np.float64)    

    for sim in prange(samples):

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

    players_won_pots = np.zeros(player_count, dtype=np.float64)

    for sim in range(samples):
        for player in range(player_count):
            players_won_pots[player] += pot_shares[sim, player]

    return {
        f"player {player + 1}:": float(players_won_pots[player] / simulations)
        for player in range(player_count)
    }

def run_exact_equity(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None = None,
    max_showdowns: int | None = 100_000,
) -> dict[str, float]:
    """
    Calculate exact equity by enumerating all compatible hand combinations
    and all possible remaining board runouts.

    Args:
        players: Validated, board-filtered player ranges.
        mode: Number of hole cards per player (4 or 5).
        board: Known community cards, or None for preflop.
        max_showdowns: Maximum estimated showdown evaluations.
            None disables the limit.

    Returns:
        Exact weighted equity for each player, as a fraction from 0 to 1.

    Raises:
        ValueError: If no compatible positive-weight matchups exist.
        RuntimeError: If the estimated workload exceeds max_showdowns.
    """
    player_count = len(players)
    board_len = 0 if board is None else len(board)
    missing_board = 5 - board_len

    board_mask = 0 if board is None else int(build_mask(board))

    # Only enumerate hands with positive weights.
    valid_indices = [
        np.flatnonzero(player.weights > 0)
        for player in players
    ]

    # Estimate an upper bound on the number of evaluations.
    remaining_cards = 52 - board_len - player_count * mode

    estimated_showdowns = (
        prod(len(indices) for indices in valid_indices)
        * comb(remaining_cards, missing_board)
    )

    if (
        max_showdowns is not None
        and estimated_showdowns > max_showdowns
    ):
        raise RuntimeError(
            f"Exact calculation exceeds the workload limit: "
            f"{estimated_showdowns:,} estimated showdowns"
        )

    selected_hands = np.empty(
        (player_count, mode),
        dtype=np.uint8,
    )

    current_board = np.empty(5, dtype=np.uint8)

    if board is not None:
        current_board[:board_len] = board

    pot_shares = np.zeros(player_count, dtype=np.float64)
    total_weight = 0.0

    # Enumerate every combination of player hands.
    for indices in product(*valid_indices):
        used_mask = board_mask
        matchup_weight = 1.0

        # Check card collisions and calculate matchup weight.
        for i, hand_index in enumerate(indices):
            player = players[i]
            hand_mask = int(player.masks[hand_index])

            if used_mask & hand_mask:
                break

            used_mask |= hand_mask
            matchup_weight *= float(player.weights[hand_index])
            selected_hands[i] = player.hands[hand_index]

        else:
            # Executed only if no collision caused a break.

            available_cards = [
                card
                for card in range(52)
                if not (used_mask & (1 << card))
            ]

            # Enumerate every possible board completion.
            # When missing_board == 0, this yields one empty tuple.
            for runout in combinations(
                available_cards,
                missing_board,
            ):
                if missing_board:
                    current_board[board_len:] = runout

                scores = numba_evaluate_hands(
                    selected_hands,
                    current_board,
                )

                highest_score = max(scores)

                winners_count = sum(
                    score == highest_score
                    for score in scores
                )

                share = matchup_weight / winners_count

                for i, score in enumerate(scores):
                    if score == highest_score:
                        pot_shares[i] += share

                total_weight += matchup_weight

    if total_weight == 0:
        raise ValueError(
            "No compatible positive-weight hand combinations exist"
        )

    return {
        f"player {i + 1}:": float(pot_shares[i] / total_weight)
        for i in range(player_count)
    }

def compute_equity(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None = None,
    samples: int = 100_000,
    seed: int = 3,
    method: Literal["exact", "monte-carlo"] = "monte-carlo",
) -> dict[str, float]:
    """Calculate equity for two to six weighted PLO4/PLO5 ranges.

    Validate inputs and remove board-blocked hands before delegating to exact
    enumeration or Monte Carlo sampling. Input ranges are not modified.

    Args:
        players: Encoded player ranges, including hands, masks, and weights.
        mode: Number of hole cards per hand (4 or 5).
        board: Known community cards (0, 3, 4, or 5), or None preflop.
        samples: Number of valid Monte Carlo showdowns; ignored for exact equity.
        seed: Reproducible Monte Carlo seed; ignored for exact equity.
        method: ``"exact"`` for exhaustive enumeration, or ``"monte-carlo"``
            for an estimated result. Exact enumeration uses its default
            workload safeguard.

    Returns:
        A mapping from player labels to equity shares in [0, 1].

    Raises:
        ValueError: If inputs are invalid or all eligible matchups are absent
            (the latter is explicitly detected by the exact method).
        RuntimeError: If the selected method exceeds its workload or attempt
            budget.
    """
    validate_equity_inputs(players, mode, board, samples, seed, method)

    valid_players = prepare_ranges(players, board)

    if method == "exact":
        return run_exact_equity(
            players=valid_players,
            mode=mode,
            board=board
        )
    
    probabilities = prepare_probabilities(valid_players)
    hands = List([p.hands for p in valid_players])
    masks = List([p.masks for p in valid_players])   
    rng = np.random.default_rng(seed)
    board_mask = np.uint64(0) if board is None else build_mask(board)
    cumulative = List()

    for p in probabilities:
        cdf = np.cumsum(p)
        cdf[-1] = 1.0
        cumulative.append(cdf)

    return run_monte_carlo(
        hands=hands,
        masks=masks,
        cumulative_probabilities=cumulative,
        mode=mode,
        board=board,
        board_mask=board_mask,
        samples=samples,
        rng=rng,
    )

