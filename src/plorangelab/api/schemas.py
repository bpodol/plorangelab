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
    player_id: int = Field(ge=0, le=5)
    hands: str
    mode: Mode


class EquityRequest(BaseModel):
    mode: Mode
    target_moe: float = Field(gt=0, le=0.005)


class EquityResponse(BaseModel):
    equities: dict[int, float]
    details: EquityDetails




