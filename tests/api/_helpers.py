"""Shared HTTP/WebSocket test builders; no fixtures or test cases."""

from fastapi.testclient import TestClient

from plorangelab.equity import EquityResult


def create_session(client: TestClient) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def add_player(
    client: TestClient,
    session_id: str,
    slot: int,
    hands: str,
) -> None:
    response = client.put(
        f"/api/sessions/{session_id}/players/{slot}",
        json={"hands": hands},
    )
    assert response.status_code == 200, response.text


def add_two_players(client: TestClient, session_id: str) -> None:
    add_player(client, session_id, 0, "AsAhQcJd")
    add_player(client, session_id, 4, "KsKhTd9c")


def equity_result(equities: dict[int, float] | None = None) -> EquityResult:
    """Small completed result with equities keyed by player slot."""
    return EquityResult(
        equities=equities if equities is not None else {0: 0.6, 4: 0.4},
        method="exact",
        margins_of_error=None,
        simulations=None,
        attempts=None,
        stop_reason=None,
        sampler=None,
        effective_samples=None,
    )


def equity_progress() -> EquityResult:
    """Representative Monte Carlo progress keyed by player slot."""
    return EquityResult(
        equities={0: 0.6, 4: 0.4},
        method="monte-carlo",
        margins_of_error={0: 0.01, 4: 0.01},
        simulations=1000,
        attempts=1000,
        stop_reason=None,
        sampler="rejection",
        effective_samples=None,
    )


def ws_url(session_id: str) -> str:
    return f"/api/sessions/{session_id}/equity/ws"


def start_message(mode: int = 4, target_moe: float = 0.002) -> dict:
    return {"type": "start", "target_moe": target_moe}
