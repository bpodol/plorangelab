"""Convert a strength-ordered CSV of PLO hands into ready-to-use NumPy arrays.

Place this script next to encoder.py, or ensure encoder is importable.
Example: python convert_ordering.py plo4_ppt_hands.csv --mode 4
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from plorangelab.encoder import _encode_hands


def convert_ordering(csv_path: Path, mode: int, output_dir: Path) -> None:
    if mode not in (4, 5):
        raise ValueError("mode must be 4 or 5")

    # Each nonempty CSV cell contains one hand, in descending strength order.
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        hands = [cell.strip() for row in csv.reader(file) for cell in row if cell.strip()]

    if not hands:
        raise ValueError("CSV contains no hands")

    for i, hand in enumerate(hands, start=1):
        if len(hand) != mode * 2:
            raise ValueError(f"Invalid hand length at position {i}: {hand!r}")

    # Reuse the existing encoder; it validates cards and duplicate hands.
    encoded = _encode_hands(hands, [1.0] * len(hands), mode)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"plo{mode}_hands.npy", encoded.hands)
    np.save(output_dir / f"plo{mode}_masks.npy", encoded.masks)
    print(f"Saved {len(hands):,} PLO{mode} hands to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--mode", type=int, choices=(4, 5), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("src/plorangelab/resources/rating"))
    args = parser.parse_args()

    convert_ordering(args.csv_path, args.mode, args.output_dir)
