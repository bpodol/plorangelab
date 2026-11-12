# Card filter expression engine

## Overview

The `plorangelab.filters` package parses card-based filter expressions, compiles them into reusable instructions, and evaluates those instructions against PLO4/PLO5 hand ranges represented by card bitmasks.

It is a **stateless evaluation engine**: given an expression and an array of hand masks, it returns a Boolean selection indicating which hands match. It does **not** modify ranges or weights, manage filter groups or activation, compute coverage, or combine a player's active filters. Those tasks belong to the separate filter-management layer.

The engine operates on **hole-card membership**, not on made-hand strength, board texture, or draw classification.

### Source files

| File | Responsibility |
| --- | --- |
| `src/plorangelab/filters/__init__.py` | Exposes the three main operations: `encode_filter`, `compile_filter`, and `evaluate_filter`. |
| `src/plorangelab/filters/models.py` | Defines the immutable expression-tree nodes and card requirements. |
| `src/plorangelab/filters/parser.py` | Parses card tokens and logical operators into an abstract syntax tree (AST). |
| `src/plorangelab/filters/compiler.py` | Converts an AST into postfix instructions and NumPy-based card-requirement tables. |
| `src/plorangelab/filters/evaluator.py` | Runs the compiled instructions against hand masks with Numba-compiled kernels. |

## Processing pipeline

```text
Filter string
    |
    v
encode_filter(expression, mode)
    |
    v
Immutable AST (AtomicFilter / And / Or / Not)
    |
    v
compile_filter(ast, mode)
    |
    v
CompiledFilter (postfix operations + card-slot masks)
    |
    v
evaluate_filter(compiled, hand_masks)
    |
    v
Boolean array aligned with the input hands
```

`evaluate_filter()` can also accept a string or AST and perform the preceding steps internally. Supplying a previously compiled program avoids reparsing and recompilation when evaluating the same expression repeatedly.

## Expression language

### Card tokens

An **atom** is a contiguous sequence of card tokens with no logical operator between them:

| Token | Meaning |
| --- | --- |
| `A`, `K`, `8` | A card of the given rank, regardless of suit. |
| `s`, `h`, `d`, `c` | A card of the given suit, regardless of rank. |
| `As`, `Kh`, `8d` | The specified exact card. |
| `AA` | Two distinct aces. |
| `ss` | Two distinct spades. |
| `Ass` | The ace of spades **and another spade**. |

Ranks use uppercase `2`–`9`, `T`, `J`, `Q`, `K`, `A`; suits use lowercase `c`, `d`, `h`, `s`. An uppercase rank immediately followed by a lowercase suit is interpreted as one exact-card token.

All requirements **within the same atom must be satisfied by distinct hole cards**. Repeated requirements are combined into a card set and a required count during parsing.

### Logical operators

| Syntax | Meaning | Precedence |
| --- | --- | --- |
| `!A` | NOT `A` (prefix negation). | Highest |
| `A:K` | `A` AND `K`. | Middle |
| `A!K` | `A` AND NOT `K` (binary exclusion). | Middle |
| `A\|K` | `A` OR `K`. | Lowest |
| `( ... )` | Explicit grouping. | Overrides precedence |

Operators at the same binary precedence level are parsed from left to right. Parentheses permit nested expressions. Whitespace may appear around operators and parentheses; tokens within an atom must remain contiguous.

**Atomic matching and logical AND are different.** For example:

- `Ass` requires `As` plus a **second, distinct** spade.
- `As:s` requires `As` and at least one spade, but **the same `As` may satisfy both separate atoms**.
- `8!88` matches a hand with at least one eight but not two or more eights.

These distinctions follow from the expression grammar and are intentionally preserved by compilation and evaluation.

## Parsing and the AST

`encode_filter(expression, mode)` uses a recursive-descent parser to construct immutable dataclass nodes:

- `CardRequirement(mask, count)` represents a number of cards required from an eligible set.
- `AtomicFilter(requirements)` groups requirements that must use distinct cards.
- `And(left, right)`, `Or(left, right)`, and `Not(child)` form logical expression trees.
- `FilterExpression` is a type alias covering these node types.

Eligible card sets are encoded as `uint64` bitmasks using the same card numbering as the range encoder. Exact-card masks contain one bit, rank masks contain four possible cards, and suit masks contain thirteen.

The parser validates the supported mode (4 or 5), expression syntax, and basic atomic constraints: an atom cannot request more cards than the mode permits, and a repeated requirement cannot request more cards than its own eligible set contains. It rejects empty expressions and unexpected trailing tokens.

The parser does **not** establish whether different overlapping requirements inside an atom can all be satisfied together. That question depends on assigning actual cards and is resolved by the evaluator.

## Compilation

`compile_filter(ast, mode)` converts the tree into **postfix (reverse Polish) instructions**. In this representation, Boolean evaluation can use a simple stack instead of traversing Python AST objects for every hand.

For example, `(A|K):!ss` compiles conceptually to:

```text
MATCH(A), MATCH(K), OR, MATCH(ss), NOT, AND
```

The compiler expands repeated counts into individual requirement **slots**. For example, `AsKK` creates one exact-`As` slot and two king slots. Slots are sorted by ascending number of eligible cards, so the matching algorithm attempts the most restrictive requirements first.

### `CompiledFilter` representation

| Field | Dtype / shape | Purpose |
| --- | --- | --- |
| `operations` | `int8`, `(instruction_count,)` | Opcodes: `MATCH`, `NOT`, `AND`, `OR`. |
| `operands` | `int32`, `(instruction_count,)` | Atom-row index for `MATCH`; `-1` for logical operations. |
| `slot_masks` | `uint64`, `(atom_count, mode)` | Eligible card masks for each atom's slots, with zero padding. |
| `slot_counts` | `int32`, `(atom_count,)` | Number of active slots per atom. |
| `mode` | `4` or `5` | PLO variant for which the program was compiled. |

The compiler marks its output NumPy arrays as read-only and stores them in a frozen, slotted dataclass. This protects against ordinary accidental modifications; it is not a security boundary or a guarantee of deep immutability.

Compilation also validates manually supplied AST requirements, including card-mask bounds, positive counts, and the mode's maximum slot count. It does not test whether the requirements are jointly satisfiable for any actual hand. The AST itself does not record its parsing mode, so compilation checks compatibility with the supplied mode rather than proving it matches a previous parse invocation.

## Evaluation

`evaluate_filter(expression, hand_masks, mode)` accepts a filter string, an AST, or a `CompiledFilter` instance. It returns a one-dimensional `bool` NumPy array with one result for each input hand and the same ordering as `hand_masks`.

The input masks must be a one-dimensional `uint64` NumPy array encoding valid PLO4/PLO5 hands, such as `EncodedRange.masks` from the range encoder. The entry point verifies the array's shape and dtype; it **assumes**, rather than independently verifies, that every mask represents a valid hand.

### Distinct-card assignment

For each atomic `MATCH`, `_matches_slots()` performs **iterative depth-first backtracking**:

1. Intersect the current hand's card mask with the next slot's eligible-card mask.
2. Choose an eligible card that has not already been assigned within the atom.
3. Continue to the next slot, tracking the occupied cards.
4. If no candidate remains, backtrack and try a different assignment.
5. Succeed only when every slot has its own distinct card.

Candidate cards are selected with bitwise operations on the lowest set bit. The slot-ordering optimization from compilation helps eliminate impossible assignments early. The maximum number of slots is bounded by the PLO mode (four or five).

### Boolean execution

`_evaluate_compiled()` processes the postfix program against each hand mask:

- `MATCH` pushes the outcome of an atomic match.
- `NOT` negates the top stack value.
- `AND` and `OR` combine the top two values.

The function allocates its Boolean stack and matching scratch buffers once per evaluation call and reuses them across hands. Both the slot matcher and the instruction interpreter use Numba's `@njit(cache=True)`. The current evaluation loop is **sequential across hand masks**; it does not use `prange`.

The returned selection does not alter the input mask array or apply any range weights.

## Public API example

```python
from plorangelab.filters import compile_filter, encode_filter, evaluate_filter

# encoded_range is an EncodedRange produced by the range encoder.
selection = evaluate_filter("Ass", encoded_range.masks, mode=4)

# Compile once when applying the same expression to several ranges.
ast = encode_filter("(AsKK|AA):!ss", mode=4)
compiled = compile_filter(ast, mode=4)

selection = evaluate_filter(compiled, encoded_range.masks)
# selection[i] indicates whether encoded_range.hands[i] matches.
```

For a precompiled program, `mode` may be omitted because it is stored in `CompiledFilter`. If a mode is explicitly supplied with a compiled program, it must agree with the program's stored mode.

## Integration boundary

The filter engine's output is a **selection mask**, not a modified `EncodedRange` or a filter-management state update. The calling layer is responsible for deciding what to do with that selection, including any combination with previously active selections, application of weights, or calculation of coverage.

This keeps the parsing/compilation/evaluation logic independent of filter trees, groups, player sessions, and activation state. Those responsibilities are handled outside `plorangelab.filters`, including by the separate filter-management subsystem.

## Design decisions and limitations

- **Reusable compiled programs:** Postfix NumPy instructions avoid Python object traversal for each evaluated hand.
- **Correct overlapping requirements:** Backtracking, rather than independent per-token checks, enforces distinct-card assignment within atoms.
- **Compact representation:** Card sets and hands use 52-bit occupancy masks stored in `uint64` arrays.
- **Small search depth:** Atomic slot counts are limited to the four or five hole cards supported by the mode.
- **Mode-specific compilation:** Programs store their mode; callers must supply compatible encoded hands.
- **No board-aware classification:** The expression language tests hole-card rank, suit, and exact-card membership, not postflop made hands, draws, or board-dependent properties.
- **No cumulative state:** Grouping, activation/deactivation, range coverage, and selection combination are outside this engine.
- **Validation boundary:** Strings and AST requirements are validated during parsing/compilation, while compiled execution assumes well-formed instructions and valid hand masks.

## Related documentation

- [`encoding.md`](encoding.md) — Card IDs, hand masks, and the `EncodedRange` representation.
- [`equity.md`](equity.md) — Equity calculations that consume weighted player ranges.
