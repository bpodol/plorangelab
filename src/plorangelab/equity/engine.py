"""Public equity API: validate inputs, choose an engine, return results."""

from collections.abc import Callable
from threading import Event
from typing import Literal

import numpy as np
from numba.typed import List
from numpy.typing import NDArray

from plorangelab.encoder import EncodedRange, Mode, build_mask
from plorangelab.rank_combination import RankLookup

from .exact import _EXACT_MAX_SHOWDOWNS, _estimate_exact_showdowns, run_exact_equity
from .importance_sampling import _run_importance_monte_carlo
from .monte_carlo import (
    _MAX_BATCH_SIZE,
    _MAX_SIMULATIONS,
    _prepare_monte_carlo,
    _probe_collision_acceptance,
    _run_batched_monte_carlo,
)
from .results import EquityResult, MonteCarloResult


def choose_method(
    board_cards: int, estimated_showdowns: int
) -> Literal["exact", "monte-carlo"]:
    if board_cards == 0:
        return "monte-carlo"

    if estimated_showdowns <= _EXACT_MAX_SHOWDOWNS:
        return "exact"

    return "monte-carlo"



def validate_equity_inputs(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None,
    seed: int,
    method: Literal["auto", "exact", "monte-carlo"],
    target_moe: float = 0.001,
    confidence_multiplier: float = 1.96,
    min_samples: int = 10_000,
    max_samples: int = _MAX_SIMULATIONS,
    batch_size: int = 10_000,

) -> None:
    """Validate the public equity API's parameters and encoded range invariants.

    Checks player count, PLO mode, board cards, seed, array shapes and dtypes,
    card uniqueness, hand-mask consistency, and strictly positive finite
    weights.
    Does not check whether a compatible multi-player matchup exists.

    Raises:
        ValueError: If any parameter, board, or encoded range is invalid.
    """
    if mode not in (4, 5):
        raise ValueError("Mode must be 4 or 5")

    if not 2 <= len(players) <= 6:
        raise ValueError("Expected 2 to 6 players")

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
            raise TypeError(f"Player {i + 1} must be an EncodedRange")

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
        
        sorted_masks = np.sort(player.masks)

        if np.any(sorted_masks[1:] == sorted_masks[:-1]):
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

    if not isinstance(method, str) or method not in ("auto", "exact", "monte-carlo"):
        raise ValueError("Method must be 'auto', 'exact', or 'monte-carlo'")

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



def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)



def compute_equity(
    players: list[EncodedRange],
    mode: Mode,
    board: NDArray[np.uint8] | None = None,
    *,
    method: Literal["auto", "exact", "monte-carlo"] = "auto",
    target_moe: float = 0.001,
    confidence_multiplier: float = 1.96,
    min_samples: int = 10_000,
    max_samples: int = _MAX_SIMULATIONS,
    batch_size: int = 10_000,
    seed: int = 3,
    cancel_event: Event | None = None,
    on_progress: Callable[[MonteCarloResult], None] | None = None,
    rank_lookup: RankLookup
) -> EquityResult:
    """Calculate equity exactly or with adaptive Monte Carlo.

    Automatic selection uses exact enumeration for postflop estimates no
    greater than ``_EXACT_MAX_SHOWDOWNS`` and adaptive Monte Carlo otherwise.
    Monte Carlo margins are approximate plug-in estimates; repeated checks and
    conditional stopping do not provide formal confidence bounds. Cancellation
    is checked between Monte Carlo batches. Exact kernels cannot be interrupted
    during execution.
    """
    if cancel_event is not None:
        if not isinstance(cancel_event, Event):
            raise ValueError("cancel_event must be a threading.Event or None")
        if cancel_event.is_set():
            raise InterruptedError(
                "Monte Carlo calculation cancelled before any simulations "
                "(before validation or preparation)"
            )

    
    validate_equity_inputs(
        players, 
        mode, 
        board, 
        seed, 
        method, 
        target_moe, 
        confidence_multiplier, 
        min_samples, 
        max_samples, 
        batch_size
    )
    
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Monte Carlo calculation cancelled before preparation")

    valid_players = prepare_ranges(players, board)
    
    if method == "auto":
        _, estimated_showdowns = _estimate_exact_showdowns(valid_players, mode, board)
        selected_method = choose_method(
            0 if board is None else len(board), estimated_showdowns
        )
    else:
        selected_method = method
    
    if selected_method == "exact":
        equities = run_exact_equity(
            players=valid_players,
            mode=mode,
            rank_lookup=rank_lookup,
            board=board,
        )
        return EquityResult(
            equities=equities,
            method="exact",
            margins_of_error=None,
            simulations=None,
            attempts=None,
            stop_reason=None,
        )

    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Monte Carlo calculation cancelled before preparation")
    hands, masks, cumulative, board_mask, rng = _prepare_monte_carlo(
        valid_players, board, seed
    )
    
    # Measure independent hand-selection acceptance only for larger
    # multiway calculations.
    use_importance = False

    if len(valid_players) >= 3 and max_samples >= 1_000:
        # Separate generator preserves the old rejection sampler's seeded output.
        probe_rng = np.random.default_rng(int(seed) + 0x9E3779B9)

        accepted = _probe_collision_acceptance(
            masks,
            cumulative,
            1_024,
            probe_rng,
        )

        use_importance = accepted < 10
  
    if use_importance:
        probabilities = List([
            player.weights.astype(np.float64)
            / player.weights.sum(dtype=np.float64)
            for player in valid_players
        ])

        result = _run_importance_monte_carlo(
            hands,
            masks,
            probabilities,
            cumulative,
            mode,
            board,
            board_mask,
            rng,
            requested_samples=max_samples,
            batch_size=batch_size,
            attempt_limit=max(10_000, max_samples * 20),
            target_moe=float(target_moe),
            min_samples=min_samples,
            confidence_multiplier=float(confidence_multiplier),
            cancel_event=cancel_event,
            on_progress=on_progress,
            rank_lookup=rank_lookup,
        )

    else:
        result = _run_batched_monte_carlo(
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
            target_moe=float(target_moe),
            min_samples=min_samples,
            confidence_multiplier=float(confidence_multiplier),
            cancel_event=cancel_event,
            on_progress=on_progress,
            rank_lookup=rank_lookup,
        )
  
    return EquityResult(
        equities=result.equities,
        method="monte-carlo",
        margins_of_error=result.margins_of_error,
        simulations=result.simulations,
        attempts=result.attempts,
        stop_reason=result.stop_reason,
        sampler=result.sampler,
        effective_samples=result.effective_samples,
    )
