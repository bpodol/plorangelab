"""Initialize shared resources used by the PLO calculation engine.

Loads precomputed PLO4/PLO5 starting-hand orderings and builds
five-card hand-ranking lookup tables.
"""

from dataclasses import dataclass

from plorangelab.rank_combination import RankLookup, build_lookup_tables
from plorangelab.rating import HandRating, load_rating
from plorangelab.types import Mode


@dataclass
class EngineResources:
    """Container for resources required by the calculation engine.

    Attributes:
        ratings: Precomputed starting-hand orderings for PLO4 and PLO5,
            used for percentage-based range selection.
        rank_lookup: Five-card hand-ranking and binomial lookup tables
            used by the equity evaluator.
    """

    ratings: dict[Mode, HandRating]
    rank_lookup: RankLookup


def load_resources() -> EngineResources:
    """Load starting-hand ratings and generate hand-ranking lookup tables.

    Loads the packaged PLO4/PLO5 hand-ordering arrays from NumPy files
    and generates the lookup tables for all possible five-card hands.

    Returns:
        EngineResources containing the loaded ratings and generated
        ranking tables.

    Notes:
        Resources are constructed on every invocation. The caller is
        responsible for retaining and reusing the returned instance.
    """
    return EngineResources(ratings=load_rating(), rank_lookup=build_lookup_tables())
