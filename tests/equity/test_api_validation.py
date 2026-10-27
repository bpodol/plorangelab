"""Public equity API input validation: modes, cards, ranges, weights, and options."""

import numpy as np
import pytest

from plorangelab.encoder import EncodedRange, build_mask, encode_board, encode_range
from plorangelab.equity import compute_equity

from ._helpers import RIVER, _guaranteed_split_players


# Player and board validation

@pytest.mark.parametrize("player_count", [1, 7])
def test_compute_equity_rejects_player_counts_outside_two_to_six(
    player_count: int,
) -> None:
    player = encode_range("AsAhKdQc", 4)

    with pytest.raises(ValueError):
        compute_equity([player] * player_count, mode=4)

def test_compute_equity_rejects_invalid_plo_mode() -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=6)  # type: ignore[arg-type]

@pytest.mark.parametrize(
    "board",
    [
        np.asarray([0, 1], dtype=np.uint8),
        np.asarray([52, 1, 2], dtype=np.uint8),
        np.asarray([0, 0, 2], dtype=np.uint8),
    ],
    ids=["invalid-length", "invalid-card-id", "duplicate-card"],
)
def test_compute_equity_rejects_invalid_boards(board: np.ndarray) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, board=board)


# Encoded range invariants

@pytest.mark.parametrize(
    "invalid_range",
    [
        EncodedRange(
            hands=encode_range("AsAhKdQc", 4).hands[0],
            masks=encode_range("AsAhKdQc", 4).masks,
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=encode_range("AsAhKdQc", 4).hands,
            masks=np.asarray([1], dtype=np.uint64),
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=encode_range("AsAhKdQc", 4).hands,
            masks=np.empty((0,), dtype=np.uint64),
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=np.asarray([[51, 51, 45, 42]], dtype=np.uint8),
            masks=encode_range("AsAhKdQc", 4).masks,
            weights=encode_range("AsAhKdQc", 4).weights,
        ),
        EncodedRange(
            hands=np.asarray([[51, 50, 45, 42], [51, 50, 45, 42]], dtype=np.uint8),
            masks=np.asarray(
                [build_mask(np.asarray([51, 50, 45, 42], dtype=np.uint8))] * 2,
                dtype=np.uint64,
            ),
            weights=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    ],
    ids=[
        "hand-dimensions",
        "incorrect-mask-content",
        "incorrect-mask-shape",
        "duplicate-cards",
        "duplicate-hands",
    ],
)
def test_compute_equity_rejects_malformed_encoded_ranges(
    invalid_range: EncodedRange,
) -> None:
    players = [invalid_range, encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, board=encode_board(RIVER))

@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
        np.asarray([-1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
    ],
    ids=["nan", "infinite", "negative", "zero-total", "non-float-dtype"],
)
def test_compute_equity_rejects_invalid_weights(weights: np.ndarray) -> None:
    valid = encode_range("AsAhKdQc", 4)
    invalid = EncodedRange(valid.hands, valid.masks, weights)
    players = [invalid, encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, board=encode_board(RIVER))

def test_compute_equity_rejects_zero_weight_hand_in_nonzero_range() -> None:
    encoded = encode_range("AsAhKdQc,KsKhJdTc", 4)
    invalid = EncodedRange(
        encoded.hands,
        encoded.masks,
        np.asarray([0.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="invalid weights"):
        compute_equity([invalid, encode_range("2s3s4s5s", 4)], mode=4)

def test_compute_equity_rejects_empty_ranges() -> None:
    empty = EncodedRange(
        hands=np.empty((0, 4), dtype=np.uint8),
        masks=np.empty((0,), dtype=np.uint64),
        weights=np.empty((0,), dtype=np.float32),
    )
    players = [empty, encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError, match="empty range"):
        compute_equity(players, mode=4)

def test_compute_equity_rejects_ranges_fully_blocked_by_board() -> None:
    players = [
        encode_range("2cAsAhKd", 4),
        encode_range("KsKhJdTd", 4),
    ]

    with pytest.raises(ValueError, match="No combinations left"):
        compute_equity(players, mode=4, board=encode_board(RIVER))


# Calculation options and limits

@pytest.mark.parametrize("max_samples", [0, -1, 1.5, True])
def test_compute_equity_rejects_invalid_sample_limits(max_samples: object) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(
            players,
            mode=4,
            max_samples=max_samples,  # type: ignore[arg-type]
            method="monte-carlo",
        )

@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_compute_equity_rejects_invalid_seeds(seed: object) -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError):
        compute_equity(players, mode=4, seed=seed)  # type: ignore[arg-type]

def test_compute_equity_rejects_invalid_computation_method() -> None:
    players = [encode_range("AsAhKdQc", 4), encode_range("KsKhJdTd", 4)]

    with pytest.raises(ValueError, match="Method"):
        compute_equity(
            players,
            mode=4,
            method="approximate",  # type: ignore[arg-type]
        )

@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_samples": 1_200_001}, "max_samples must not exceed"),
        ({"batch_size": 100_001}, "batch_size must not exceed"),
    ],
)
def test_monte_carlo_rejects_resource_limits(
    kwargs: dict[str, int], message: str
) -> None:
    players, board = _guaranteed_split_players()

    with pytest.raises(ValueError, match=message):
        compute_equity(players, mode=4, board=board, method="monte-carlo", **kwargs)
