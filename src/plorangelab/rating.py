"""Load precomputed PLO4/PLO5 hand-rating data from package resources.

The rating data provides ordered hand combinations and their card masks
for percentage-based range selection.
"""

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray

from plorangelab.types import Mode

ratings_dir = files("plorangelab").joinpath("resources", "rating")


@dataclass
class HandRating:
    """Precomputed hand combinations and their corresponding card masks.

    Attributes:
        hands: Two-dimensional uint8 array of encoded PLO hands, ordered
            according to the precomputed rating data.
        masks: One uint64 card mask per hand, aligned with the hands array.
    """

    hands: NDArray[np.uint8]
    masks: NDArray[np.uint64]


def load_rating() -> dict[Mode, HandRating]:
    """Load the packaged PLO4 and PLO5 hand-rating arrays.

    Reads precomputed hand and mask arrays from NumPy .npy files
    located in the package's resources/rating directory.

    Returns:
        A dictionary mapping each PLO mode (4 or 5) to its HandRating
        instance containing the corresponding hands and masks.

    Raises:
        OSError: If a required resource file cannot be opened.
        ValueError: If a resource file contains invalid NumPy data.
    """
    ratings = {}

    for mode in (4, 5):
        with ratings_dir.joinpath(f"plo{mode}_hands.npy").open("rb") as f:
            hands = np.load(f)

        with ratings_dir.joinpath(f"plo{mode}_masks.npy").open("rb") as f:
            masks = np.load(f)

        ratings[mode] = HandRating(hands=hands, masks=masks)

    return ratings
