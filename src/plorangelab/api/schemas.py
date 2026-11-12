"""Define Pydantic request and response models for the API.

Provides validation and serialization contracts for session,
player range, board, filter, and equity operations.

Models are shared between REST endpoints and WebSocket
message construction where applicable.
"""

from typing import Literal

from pydantic import BaseModel, Field

from plorangelab.equity.results import StopReason
from plorangelab.types import Mode


class BoardRequest(BaseModel):
    """Request to update or clear the shared community board.

    A null board clears the current board and returns players to
    preflop. Non-null board strings are validated and encoded by
    the range-management layer.
    """

    board: str | None


class EquityDetails(BaseModel):
    """Describe the execution method and diagnostics of an equity result.

    Includes the calculation method, optional Monte Carlo statistics,
    sampling information, and the reason execution stopped.

    Fields that do not apply to a particular calculation method
    are represented by None.
    """

    method: Literal["exact", "monte-carlo"]
    margins_of_error: list[float | None] | None
    simulations: int | None
    attempts: int | None
    stop_reason: StopReason | None
    sampler: Literal["rejection", "importance"] | None
    effective_samples: float | None


class RangeRequest(BaseModel):
    hands: str
    mode: Mode


class EquityRequest(BaseModel):
    """Specify the PLO mode and requested equity calculation precision.

    The target margin of error must be greater than zero and at
    most 0.005. It is used by the equity engine when performing
    Monte Carlo calculations.
    """

    mode: Mode
    target_moe: float = Field(gt=0, le=0.005)


class EquityResponse(BaseModel):
    """Return calculated equities mapped to their player slot indices.

    The equities dictionary contains only participating players,
    using their zero-based slot indices as keys.

    Execution details describe the calculation method and
    available diagnostics.
    """

    equities: dict[int, float]
    details: EquityDetails


class BoardResponse(BaseModel):
    """Return current hand counts for all six player slots.

    The players list follows slot order. Occupied slots contain
    their current range size, while empty slots contain zero.
    """

    players: list[int]


class RangeResponse(BaseModel):
    hands: int


class SessionResponse(BaseModel):
    session_id: str


class FilterCreateRequest(BaseModel):
    expression: str = Field(min_length=1)
    name: str | None = None
    path: tuple[str, ...] = ()


class GroupCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    path: tuple[str, ...] = ()


class GroupResponse(BaseModel):
    name: str
    path: tuple[str, ...]


class SetActiveRequest(BaseModel):
    active: bool


class FilterCoverageResponse(BaseModel):
    """Return weighted range coverage after a filter operation.

    For filter creation, coverage describes the new filter's
    membership relative to the active street's base range.

    For activation, deletion, and clearing operations, coverage
    describes the resulting effective range.
    """

    coverage: float


class FilterInfoResponse(BaseModel):
    type: Literal["filter"]
    name: str
    expression: str
    coverage: float
    active: bool


class FilterGroupInfoResponse(BaseModel):
    """Represent a filter group and its recursively nested children.

    The filters dictionary maps child names to either individual
    filter metadata or nested group metadata.

    The type field distinguishes groups from individual filters
    when interpreting the serialized tree.
    """

    type: Literal["group"]
    name: str
    coverage: float
    active: bool
    filters: dict[str, FilterInfoResponse | FilterGroupInfoResponse]
