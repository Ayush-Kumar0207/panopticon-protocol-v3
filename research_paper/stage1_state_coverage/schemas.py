from pydantic import BaseModel
from typing import Optional, List

# Version strings required by section 8.1
INPUT_VERSION = "panopticon-stage1-synthetic-input-v1"
TURN_METRICS_VERSION = "panopticon-stage1-turn-metrics-v1"
REPRO_MANIFEST_VERSION = "panopticon-stage1-repro-manifest-v1"

class TurnMetric(BaseModel):
    schema_version: str = TURN_METRICS_VERSION
    synthetic: bool = True
    episode_id: str
    level: str
    turn: int
    parse_success: bool
    raw_semantic_valid: bool
    executed_semantic_valid: bool
    oracle_legal: Optional[bool] = None
    intervention_applied: bool
