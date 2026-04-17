"""
OpenEnv Starter Kit — Pydantic Data Models
=============================================

Define your environment's state, actions, and observations here.
All models use Pydantic v2 for strict JSON serialization.

TODO: Replace the example Enums and Models with your problem domain.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMERATIONS — Define your discrete value spaces
# =============================================================================

class EntityState(str, Enum):
    """State of an entity in your environment. TODO: Customize."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROCESSING = "processing"
    FAILED = "failed"

class EntityTier(int, Enum):
    """Priority/criticality tier. TODO: Customize."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class ActionType(str, Enum):
    """Available agent actions. TODO: Customize."""
    INSPECT = "inspect"
    PROCESS = "process"
    REPAIR = "repair"
    WAIT = "wait"
    NOOP = "noop"


# =============================================================================
# ENTITY MODELS — Your environment's objects
# =============================================================================

class Entity(BaseModel):
    """A single entity in the environment. TODO: Customize fields."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Unique entity identifier")
    name: str = Field(description="Human-readable name")
    state: EntityState = Field(default=EntityState.ACTIVE, description="Current state")
    tier: EntityTier = Field(default=EntityTier.MEDIUM, description="Priority tier")
    score: float = Field(default=0.0, description="Entity-specific score/metric")


class Task(BaseModel):
    """A task/objective to be completed. TODO: Customize fields."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Task identifier")
    description: str = Field(default="", description="Task description")
    priority: float = Field(default=5.0, description="Priority score (higher = more urgent)")
    target_entities: list[str] = Field(default_factory=list, description="Entity IDs this task applies to")
    completed: bool = Field(default=False)


class Relationship(BaseModel):
    """A relationship/dependency between entities. TODO: Customize."""
    source: str = Field(description="Source entity ID")
    target: str = Field(description="Target entity ID")
    relationship_type: str = Field(default="depends_on")


# =============================================================================
# ACTION & OBSERVATION — The agent interface
# =============================================================================

class AgentAction(BaseModel):
    """Action the agent can take each turn. TODO: Customize."""
    model_config = ConfigDict(use_enum_values=True)

    action_type: ActionType = Field(description="Type of action to perform")
    target: str = Field(default="", description="Target entity ID")
    task_id: str | None = Field(default=None, description="Associated task ID (if applicable)")
    reason: str = Field(default="", description="Optional reasoning for the action")


class EnvironmentObservation(BaseModel):
    """What the agent sees each turn. TODO: Customize."""
    model_config = ConfigDict(use_enum_values=True)

    entities: list[Entity] = Field(default_factory=list, description="All entities")
    tasks: list[Task] = Field(default_factory=list, description="Active tasks/objectives")
    relationships: list[Relationship] = Field(default_factory=list, description="Entity relationships")
    turn: int = Field(default=0, description="Current turn number")
    max_turns: int = Field(default=50, description="Maximum turns allowed")
    messages: list[str] = Field(default_factory=list, description="System messages")


class EnvironmentState(BaseModel):
    """Full internal state (superset of observation, for grading). TODO: Customize."""
    model_config = ConfigDict(use_enum_values=True)

    entities: list[Entity] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    turn: int = Field(default=0)
    max_turns: int = Field(default=50)
    total_reward: float = Field(default=0.0)
    done: bool = Field(default=False)


# =============================================================================
# HELPERS
# =============================================================================

def validate_action(action: AgentAction, obs: EnvironmentObservation) -> tuple[bool, str]:
    """Validate an action against the current observation. TODO: Customize."""
    entity_ids = {e.id for e in obs.entities}

    if action.action_type == ActionType.NOOP:
        return True, "NOOP is always valid"

    if action.target and action.target not in entity_ids:
        return False, f"Unknown target entity: {action.target}"

    return True, "Valid action"
