"""Tests for application resource initialization."""

from unittest.mock import Mock

import pytest
from plorangelab.bootstrap import EngineResources

from plorangelab import bootstrap


def test_load_resources_initializes_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fake resources, avoiding expensive real initialization.
    ratings = {4: object(), 5: object()}
    rank_lookup = object()

    mock_load_rating = Mock(return_value=ratings)
    mock_build_lookup = Mock(return_value=rank_lookup)

    monkeypatch.setattr(bootstrap, "load_rating", mock_load_rating)
    monkeypatch.setattr(bootstrap, "build_lookup_tables", mock_build_lookup)

    result = bootstrap.load_resources()

    assert isinstance(result, EngineResources)
    assert result.ratings is ratings
    assert result.rank_lookup is rank_lookup

    mock_load_rating.assert_called_once_with()
    mock_build_lookup.assert_called_once_with()