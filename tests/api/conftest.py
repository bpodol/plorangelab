"""FastAPI test application and lifespan fixture."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from plorangelab.api import main as api_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    resources = SimpleNamespace(
        ratings={4: object(), 5: object()},
        rank_lookup=object(),
    )
    load_resources = Mock(return_value=resources)
    monkeypatch.setattr(api_module, "load_resources", load_resources)

    # Runs the real FastAPI lifespan and its bounded ThreadPoolExecutor.
    with TestClient(api_module.app) as test_client:
        load_resources.assert_called_once_with()
        yield test_client
