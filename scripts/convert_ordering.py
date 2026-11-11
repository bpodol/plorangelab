"""Convert preordered PLO hand ranges from CSV into NumPy lookup arrays.

Reads PLO4 or PLO5 hands from a CSV file already ordered by preflop
strength, encodes them using the project's range encoder, and saves
the resulting hand and card-mask arrays as .npy resources.

The script preserves the source ordering without calculating or
validating hand strength.

Usage:
    uv run python scripts/convert_ordering.py plo4_ppt_hands.csv --mode 4
    uv run python scripts/convert_ordering.py plo5_ppt_hands.csv --mode 5

Output:
    plo{mode}_hands.npy: Encoded hands in the original CSV order.
    plo{mode}_masks.npy: Corresponding uint64 card masks.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from plorangelab.encoder import _encode_hands


def convert_ordering(csv_path: Path, mode: int, output_dir: Path) -> None:
    """Convert a strength-ordered CSV into encoded hand and mask arrays.

    Reads all nonempty CSV cells in their original order and validates
    hand lengths, card representations, and duplicate combinations
    using the shared encoder.

    Creates the output directory if necessary and saves two NumPy files
    with aligned entries, preserving the source ordering.

    Args:
        csv_path: Path to a CSV containing preordered PLO hands.
        mode: Number of hole cards per hand (4 or 5).
        output_dir: Directory where the .npy files will be saved.

    Raises:
        ValueError: If the mode is invalid, the CSV contains no hands,
            or any hand fails validation.
        OSError: If an input file cannot be read or output files
            cannot be written.
    """
    if mode not in (4, 5):
        raise ValueError("mode must be 4 or 5")

    # Each nonempty CSV cell contains one hand, in descending strength order.
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        hands = [
            cell.strip() for row in csv.reader(file) for cell in row if cell.strip()
        ]

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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("src/plorangelab/resources/rating")
    )
    args = parser.parse_args()

    convert_ordering(args.csv_path, args.mode, args.output_dir)
