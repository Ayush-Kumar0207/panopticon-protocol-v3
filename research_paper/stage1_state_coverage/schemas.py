from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal, Optional

# Version strings required by section 8.1
INPUT_VERSION = "panopticon-stage1-synthetic-input-v1"
TURN_METRICS_VERSION = "panopticon-stage1-turn-metrics-v1"
REPRO_MANIFEST_VERSION = "panopticon-stage1-repro-manifest-v1"

class TurnMetric(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    schema_version: Literal[TURN_METRICS_VERSION]
    synthetic: bool
    episode_id: str = Field(min_length=1)
    level: Literal["easy", "medium", "hard", "level_4", "level_5"]
    turn: int = Field(ge=0)
    parse_success: bool
    raw_semantic_valid: bool
    executed_semantic_valid: bool
    oracle_legal: Optional[bool] = None
    intervention_applied: bool

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("episode_id must be nonblank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_flags(self):
        if (
            not self.intervention_applied
            and self.raw_semantic_valid != self.executed_semantic_valid
        ):
            raise ValueError(
                "raw/executed validity mismatch requires intervention"
            )
        return self
