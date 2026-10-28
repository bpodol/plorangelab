import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Literal
from importlib.resources import files


Mode = Literal[4, 5]


rankings_dir = files("plorangelab").joinpath("resources", "rating")

@dataclass
class HandsRating:
    hands: NDArray[np.uint8]
    masks: NDArray[np.uint64]


def load_rating() -> dict[Mode, HandsRating]:
    ratings = {}

    for mode in (4, 5):
        with rankings_dir.joinpath(f"plo{mode}_hands.npy").open("rb") as f:
            hands = np.load(f)

        with rankings_dir.joinpath(f"plo{mode}_masks.npy").open("rb") as f:
            masks = np.load(f)

        ratings[mode] = HandsRating(hands=hands, masks=masks)

    return ratings