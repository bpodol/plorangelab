from dataclasses import dataclass
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray

from plorangelab.types import Mode

ratings_dir = files("plorangelab").joinpath("resources", "rating")

@dataclass
class HandRating:
    hands: NDArray[np.uint8]
    masks: NDArray[np.uint64]


def load_rating() -> dict[Mode, HandRating]:
    ratings = {}

    for mode in (4, 5):
        with ratings_dir.joinpath(f"plo{mode}_hands.npy").open("rb") as f:
            hands = np.load(f)

        with ratings_dir.joinpath(f"plo{mode}_masks.npy").open("rb") as f:
            masks = np.load(f)

        ratings[mode] = HandRating(hands=hands, masks=masks)

    return ratings