"""Versioned schemas for authorized, long-horizon agent-security campaigns."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "panopticon-bench-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InterventionLevel(str, Enum):
    RAW = "raw"
    PARSE_CLEANUP = "parse_cleanup"
    SEMANTIC_REPAIR = "semantic_repair"
    SUPERVISOR = "supervisor"
    HYBRID = "hybrid"


class TaxonomyReference(StrictModel):
    framework: str = Field(min_length=2, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    identifier: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=2, max_length=160)
    url: str | None = None


class CampaignStep(StrictModel):
    step_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    phase: str = Field(min_length=2, max_length=80)
    prompt: str = Field(min_length=1, max_length=20_000)
    expected_safe_behavior: str = Field(min_length=2, max_length=2_000)
    wait_turns: int = Field(default=0, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CampaignSpec(StrictModel):
    schema_version: str = SCHEMA_VERSION
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    title: str = Field(min_length=4, max_length=200)
    description: str = Field(min_length=10, max_length=4_000)
    authorization_required: bool = True
    destructive: bool = False
    max_turns: int = Field(ge=1, le=500)
    synthetic_canaries: list[str] = Field(default_factory=list, max_length=20)
    taxonomy: list[TaxonomyReference] = Field(default_factory=list)
    steps: list[CampaignStep] = Field(min_length=1, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("synthetic_canaries")
    @classmethod
    def validate_canaries(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("synthetic canaries must be unique")
        for value in values:
            if len(value) < 8 or len(value) > 256:
                raise ValueError("synthetic canaries must contain 8-256 characters")
        return values

    @model_validator(mode="after")
    def enforce_safe_defaults(self) -> "CampaignSpec":
        if self.destructive:
            raise ValueError("v1 campaign files must be non-destructive")
        if not self.authorization_required:
            raise ValueError("v1 campaigns must require explicit authorization")
        if len(self.steps) + sum(step.wait_turns for step in self.steps) > self.max_turns:
            raise ValueError("steps plus wait turns exceed max_turns")
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign step IDs must be unique")
        return self


class ToolCall(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any] = Field(default_factory=dict)


class TargetRequest(StrictModel):
    session_id: str = Field(min_length=3, max_length=200)
    turn: int = Field(ge=0)
    content: str = Field(max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetResponse(StrictModel):
    content: str = Field(default="", max_length=200_000)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepAssessment(StrictModel):
    attack_success: bool = False
    operational_pass: bool = True
    constraint_failures: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)
    utility_score: float = Field(default=1.0, ge=0.0, le=1.0)


class EventRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    experiment_id: str
    campaign_id: str
    campaign_version: str
    target_id: str
    session_id: str
    step_id: str
    turn: int = Field(ge=0)
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_excerpt: str | None = Field(default=None, max_length=2_000)
    attack_success: bool
    operational_pass: bool
    constraint_failures: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)
    utility_score: float = Field(ge=0.0, le=1.0)
    intervention_level: InterventionLevel = InterventionLevel.RAW
    intervention_used: bool = False
    intervention_changed_action: bool = False
    latency_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    campaign_id: str
    campaign_version: str
    target_id: str
    session_id: str
    completed: bool
    turns: int = Field(ge=0)
    composite_score: float = Field(ge=0.0, le=1.0)
    composite_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    composite_pass: bool
    operational_pass: bool
    attack_success: bool
    constraint_failures: list[str] = Field(default_factory=list)
    intervention_level: InterventionLevel = InterventionLevel.RAW
    intervention_used: bool = False
    intervention_changed_outcome: bool = False
    utility_score: float = Field(ge=0.0, le=1.0)
    baseline_utility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    time_to_compromise: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
