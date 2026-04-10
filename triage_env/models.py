from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal[
    "assign_low_priority",
    "assign_medium_priority",
    "assign_high_priority",
    "send_to_emergency",
    "request_additional_tests",
]

InjurySeverity = Literal["low", "medium", "high"]


class Observation(BaseModel):
    patient_id: int = Field(..., description="Unique patient identifier")
    symptoms: str = Field(..., description="Short symptom summary")
    heart_rate: int = Field(..., ge=20, le=250)
    blood_pressure: str = Field(..., description="Blood pressure in systolic/diastolic format")
    injury_severity: InjurySeverity
    waiting_time: int = Field(..., ge=0, description="Minutes waiting in triage")


class Action(BaseModel):
    action: ActionType


class Reward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
    reason: str
    priority_correct: bool
    emergency_handling_correct: bool
    delay_penalty_applied: bool
    wrong_decision_penalty_applied: bool
