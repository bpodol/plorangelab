from dataclasses import dataclass

from plorangelab.rank_combination import RankLookup, build_lookup_tables
from plorangelab.rating import HandRating, load_rating
from plorangelab.types import Mode


@dataclass
class EngineResources:
    ratings: dict[Mode, HandRating]
    rank_lookup: RankLookup


def load_resources():
    return EngineResources(ratings=load_rating(), rank_lookup=build_lookup_tables())
