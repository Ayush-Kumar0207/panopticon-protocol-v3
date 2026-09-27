from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal, Optional

# Version strings required by section 8.1
INPUT_VERSION = "panopticon-stage1-synthetic-input-v1"
TURN_METRICS_VERSION = "panopticon-stage1-turn-metrics-v2"
REPRO_MANIFEST_VERSION = "panopticon-stage1-repro-manifest-v1"

class TurnMetric(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    schema_version: Literal[TURN_METRICS_VERSION]
    synthetic: bool

    experiment_id: str = Field(min_length=1)
    run_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    checkpoint_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    source_commit: str = Field(
        min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$"
    )
    feature_extractor_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    evidence_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    evidence_bytes: int = Field(gt=0)
    episode_id: str = Field(min_length=1)
    level: Literal["easy", "medium", "hard", "level_4", "level_5"]
    turn: int = Field(ge=0)
    parse_success: bool
    raw_semantic_valid: bool
    executed_semantic_valid: bool
    oracle_legal: Optional[bool] = None
    intervention_applied: bool

    @field_validator(
        "experiment_id", "feature_extractor_version", "episode_id"
    )
    @classmethod
    def validate_nonblank_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identity must be nonblank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_flags(self):
        if not self.parse_success and self.raw_semantic_valid:
            raise ValueError(
                "parse_failed_raw_cannot_be_semantically_valid"
            )
        if (
            not self.intervention_applied
            and self.raw_semantic_valid != self.executed_semantic_valid
        ):
            raise ValueError(
                "raw/executed validity mismatch requires intervention"
            )
        return self
