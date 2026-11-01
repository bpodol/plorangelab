"""Session-scoped lookup shared by the equity test suite."""

import pytest
from plorangelab.rank_combination import RankLookup, build_lookup_tables


@pytest.fixture(scope="session")
def rank_lookup() -> RankLookup:
    """Generate the rank and binomial tables once per test session."""
    return build_lookup_tables()
