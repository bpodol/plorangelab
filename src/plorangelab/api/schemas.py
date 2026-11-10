from typing import Literal

from pydantic import BaseModel, Field

from plorangelab.equity.results import StopReason
from plorangelab.types import Mode


class BoardRequest(BaseModel):
    board: str | None


class EquityDetails(BaseModel):
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
    mode: Mode
    target_moe: float = Field(gt=0, le=0.005)


class EquityResponse(BaseModel):
    equities: dict[int, float]
    details: EquityDetails


class BoardResponse(BaseModel):
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
    coverage: float


class FilterInfoResponse(BaseModel):
    type: Literal["filter"]
    name: str
    expression: str
    coverage: float
    active: bool


class FilterGroupInfoResponse(BaseModel):
    type: Literal["group"]
    name: str
    coverage: float
    active: bool
    filters: dict[str, FilterInfoResponse | FilterGroupInfoResponse]
