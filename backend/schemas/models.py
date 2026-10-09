"""
API Request and Response Data Schemas.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool = True
    elkai: bool = True
    tsplib95: bool = True
    version: str = "4.2"


class ExperimentStatus(BaseModel):
    generation: int
    best_cost: float
    gap: Optional[float]
    best_generation: int
    no_improvement: int
    stagnation_patience: int
    stagnation_display: str
    converged: bool
    is_stopped: bool
    state: str
    stop_reason: str
    population_size: int
    eta_seconds: Optional[float]
    eta_text: str
    lkh_status: str
    lkh_calls_total: int
