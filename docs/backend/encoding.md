# Card and range encoding

## Overview

The encoding subsystem converts human-readable PLO4 and PLO5 card and range notation into compact data structures for equity calculation and filtering. It also loads precomputed starting-hand orderings used to interpret percentage-based ranges.

The subsystem **does not calculate preflop hand strength**. Percentage ranges select slices of an externally supplied ordering.

### Source files

| File | Responsibility |
| --- | --- |
| `src/plorangelab/types.py` | Defines `Mode` as `Literal[4, 5]`. |
| `src/plorangelab/encoder.py` | Encodes cards, boards, and weighted ranges; decodes cards and hands; selects percentage ranges. |
| `src/plorangelab/rating.py` | Loads packaged PLO4/PLO5 starting-hand orderings and their card masks. |
| `scripts/convert_ordering.py` | Converts an already ordered CSV into the NumPy resources loaded by `rating.py`. |

## Card representation

Every card is represented by an integer ID from `0` to `51`:

```text
card_id = 4 * rank_index + suit_index
```

Ranks are indexed from `2 = 0` through `A = 12`. Suits are encoded as `c = 0`, `d = 1`, `h = 2`, and `s = 3`.

Examples:

| Card | Encoded ID |
| --- | ---: |
| `2c` | 0 |
| `2s` | 3 |
| `Ac` | 48 |
| `As` | 51 |

Hands and boards use NumPy `uint8` arrays. The encoder sorts cards **within each encoded hand** by descending card ID; board input retains its supplied order.

### Card masks

`build_mask()` represents a collection of cards as a `uint64` bitmask, setting the bit at each card's ID. A bitwise AND identifies overlap between two masks:

```python
has_overlap = bool(mask_a & mask_b)
```

A mask also identifies a hand independently of its card order, allowing the bulk encoder to detect duplicate combinations.

## Encoded ranges

The `EncodedRange` dataclass groups three index-aligned NumPy arrays:

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `hands` | `(n, mode)` | `uint8` | Encoded PLO4 or PLO5 hands, with each row sorted descending. |
| `masks` | `(n,)` | `uint64` | One card-occupancy bitmask per hand. |
| `weights` | `(n,)` | `float32` | Relative positive weight for each hand. |

The dataclass is a container rather than a validator. Callers must preserve array alignment; the public equity entry point validates its encoded range inputs.

### Explicit weighted ranges

`encode_range()` accepts comma-separated, fully specified hands with an optional `@` weight suffix:

```text
AsKsQhJh@100,AdKdQcJc@50
```

The example is for PLO4. An omitted suffix defaults to full weight (`1.0`). Explicit suffixes are percentages converted to fractions; `@50` becomes `0.5` and `@100` becomes `1.0`. Fractional positive percentages are supported as well. Invalid cards, repeated cards within a hand, duplicate hands, invalid hand lengths, and nonpositive or nonfinite weights are rejected.

The bulk encoding path packs hand notation into an ASCII byte array, then uses a Numba-compiled kernel to decode, sort, and construct masks. The kernel returns status codes to its Python wrapper, which raises informative `ValueError` exceptions for invalid input.

### Board encoding

`encode_board()` accepts exactly three, four, or five concatenated two-character cards (flop, turn, or river), returning a one-dimensional `uint8` array. Duplicate cards and invalid notation are rejected. The preflop state is represented separately (for example, by `None` in the equity API), rather than by calling `encode_board()` with an empty string.

## Percentage-based ranges

`encode_input()` dispatches to either explicit range parsing or percentage-based selection. An input containing `%` is treated as a percentage expression and requires the supplied `HandRating` ordering.

| Input | Meaning |
| --- | --- |
| `50%` | First 50% of hands in the supplied ordering. |
| `50%-7%` | Hands from the 7% boundary up to the 50% boundary. |
| `10.5%` | First 10.5% of hands in the ordering. |

For an ordering of length `n`, boundaries are converted to indices by truncation:

```text
start = int(n * lower_percent / 100)
end   = int(n * upper_percent / 100)
selected = ordering[start:end]
```

Selected hands inherit the ordering's encoded cards and masks, and all receive a weight of `1.0`. An interval producing no hands is rejected.

**Interpretation:** `50%` means the first half of the **imported ordering**. It is not a claim of 50% equity, a position-specific preflop strategy, or a solver-derived range. This implementation also does not accept PPT's full general-purpose range-expression language: its percentage parser supports the forms shown above.

## Starting-hand ordering resources

`rating.py` loads the following packaged NumPy resources for each supported mode:

```text
src/plorangelab/resources/rating/
    plo4_hands.npy
    plo4_masks.npy
    plo5_hands.npy
    plo5_masks.npy
```

`load_rating()` returns a dictionary mapping mode (`4` or `5`) to `HandRating(hands, masks)` instances. The hand order is semantically significant: the first rows are intended to be the strongest according to the imported source ranking. Masks and hands must remain aligned.

### Data provenance: ProPokerTools

The project's starting-hand orderings were sourced from publicly available **ProPokerTools (PPT)** hand-ordering data and transformed into the NumPy format described above. PPT's historical ordering methodology is described by its author in the references below.

PPT originally considered ranking hands by equity versus random opponents. It then developed an **evolutionary simulation**: candidate starting hands were repeatedly evaluated against opponents increasingly restricted to hands that performed profitably in earlier iterations. This was designed to give an ordering more representative of competition against playable hands, rather than only random-hand equity.

PPT published distinct hand orderings for different table sizes, including full-ring and six-handed variants, for both four-card and five-card Omaha High.

PLO RangeLab uses the six-handed (6-max) ProPokerTools hand orderings for both PLO4 and PLO5. These rankings are converted into NumPy lookup arrays and packaged with the application. The application loads one fixed ordering per PLO mode.

These ratings are **not** equivalent to the five-card showdown-ranking lookup generated by `rank_combination.py`. Starting-hand ordering determines which combinations a percentage selects; the showdown lookup ranks made five-card poker hands.

### Converting source CSVs

`scripts/convert_ordering.py` reads every nonempty CSV cell in row order, validates the hands through the shared encoder, and writes aligned arrays without changing the supplied hand order.

Run from the repository root, using the appropriate already ordered source CSV:

```bash
uv run python scripts/convert_ordering.py path/to/plo4_ordering.csv --mode 4
uv run python scripts/convert_ordering.py path/to/plo5_ordering.csv --mode 5
```

By default, the files are saved to `src/plorangelab/resources/rating/`. Pass `--output-dir` to write elsewhere. **Existing output files with the same names will be overwritten.**

The converter validates notation and duplicate combinations but **does not verify ranking methodology, strength order, or coverage of every possible PLO combination**. Reproducibility therefore also depends on retaining the source CSVs or reliable links, their ordering variant, and the method by which they were obtained.

### Attribution and redistribution

**Starting-hand ordering data: ProPokerTools (PPT).** The credit applies to the imported ranking data, not to PLO RangeLab's encoding implementation.

In a June 2012 forum discussion, PPT's developer indicated that he preferred making the rankings available with credit to ProPokerTools. This historical statement is useful provenance, but it is **not a verified formal license** for these exact files or their redistribution in a commercial product. Before distributing the converted tables publicly or commercially, confirm the applicable source terms and preserve attribution. No endorsement by ProPokerTools is implied.

## Assumptions and limitations

- Only **PLO4 and PLO5** are supported by this encoding module.
- Individual card IDs are in `0..51`; mask operations assume a standard 52-card deck.
- Literal hands contain exactly `mode` cards; board strings contain exactly three, four, or five cards.
- Percentage boundaries depend on the **number and order of rows in the loaded source arrays**.
- The encoded arrays and `HandRating` objects do not automatically validate their contents when constructed directly.
- The original PPT ordering variant and dataset-specific reuse terms should be verified and recorded before publishing the source ranking tables.

## References

- [Poker Coder — Towards a Meaningful Ordering of Hands (2006)](https://pokercoder.blogspot.com/2006/07/towards-meaningful-ordering-of-hands.html) — Original explanation of the evolutionary ranking approach.
- [Poker Coder — 6-Handed Hand Orderings (2012)](https://pokercoder.blogspot.com/2012/03/6-handed-hand-orderings.html) — Discussion of table-size-specific orderings and downloadable data.
- [Two Plus Two — ProPokerTools/Slice discussion (2012)](https://forumserver.twoplustwo.com/38/omaha/s-propokertools-slice-discussion-thread-1203415/) — Developer's public discussion of the ordering process and attribution.
- [ProPokerTools — Range of Hands](https://www.propokertools.com/oracle_help/range_of_hands) — Historical source page for percentage-ordering data (availability may vary).
