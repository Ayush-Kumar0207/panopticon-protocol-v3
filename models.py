"""
The Panopticon Protocol v3 — Pydantic Data Models
====================================================

Counter-espionage RL environment where ARGUS (the agent) defends a
corporate network against HYDRA (adaptive adversary) using canary traps,
double agents, and disinformation campaigns.

"Among Us… for AIs."

All models use Pydantic v2 for strict JSON serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMERATIONS
# =============================================================================

class WorkerState(str, Enum):
    """Observable state of a worker in the enterprise."""
    LOYAL = "loyal"                    # Genuine employee, contributing
    SUSPECTED = "suspected"            # Under investigation by ARGUS
    TERMINATED = "terminated"          # Fired / removed from network
    DOUBLE_AGENT = "double_agent"      # Turned sleeper — now works for ARGUS
    COMPROMISED = "compromised"        # Publicly burned (cover blown)


class HiddenWorkerState(str, Enum):
    """Hidden ground-truth state — INVISIBLE to the agent's observation."""
    CLEAN = "clean"                    # Genuinely loyal
    SLEEPER_DORMANT = "dormant"        # Infiltrated but not yet active
    SLEEPER_ACTIVE = "active"          # Actively leaking intelligence
    TURNED = "turned"                  # Converted to double agent by ARGUS
    DEAD_SWITCH_ARMED = "dead_switch"  # Gen-4: will damage revenue if terminated


class Department(str, Enum):
    """Corporate departments — each with distinct strategic value."""
    ENGINEERING = "engineering"        # Core product, high revenue impact
    FINANCE = "finance"               # Budget access, moderate value
    RD = "rd"                         # R&D: innovation pipeline, canary-rich
    OPERATIONS = "operations"         # Logistics, low-profile
    EXECUTIVE = "executive"           # C-suite, maximum damage if compromised
    LEGAL = "legal"                   # Compliance, moderate value


class LeakChannel(str, Enum):
    """External channels where leaked intel surfaces."""
    MARKET_CHATTER = "market_chatter"       # Analyst forums, easy to monitor
    DARK_WEB = "dark_web"                   # Underground markets, hard to trace
    COMPETITOR_FILING = "competitor_filing"  # Patent/SEC filings by rivals
    PRESS_LEAK = "press_leak"               # Journalist tipoffs
    INSIDER_TRADE = "insider_trade"         # Suspicious stock movements


class SleeperGeneration(int, Enum):
    """Sleeper sophistication — each generation is harder to detect."""
    GEN_1 = 1  # Amateur: obvious leaking, caught by basic canary
    GEN_2 = 2  # Professional: avoids canary departments, needs correlation
    GEN_3 = 3  # Spy: plants false evidence framing innocent workers
    GEN_4 = 4  # Cell operative: dead-man's switch damages revenue on termination
    GEN_5 = 5  # Manchurian: appears as top performer, acts only at executive level


class ActionType(str, Enum):
    """
    8 streamlined agent actions — PPO-friendly.
    Merges the original 12+ types into clean categories.
    """
    # ── PRODUCTIVITY (2) ──
    WORK = "work"           # Process project workflow in a department
    HIRE = "hire"           # Replace a terminated worker

    # ── INTELLIGENCE (3) ──
    CANARY = "canary"       # Inject canary trap into a department
    MONITOR = "monitor"     # Check external leak channels
    INVESTIGATE = "investigate"  # Audit worker / verify leak / correlate signals

    # ── ENFORCEMENT (2) ──
    NEUTRALIZE = "neutralize"     # Terminate / interrogate / turn a suspect
    DEPLOY_DOUBLE = "deploy_double"  # Feed disinfo through turned asset

    # ── META ──
    NOOP = "noop"           # Skip turn


class SubAction(str, Enum):
    """Sub-action modifier for INVESTIGATE and NEUTRALIZE."""
    NONE = "none"
    # INVESTIGATE sub-actions
    AUDIT = "audit"         # Deep-audit a specific worker's activity
    VERIFY = "verify"       # Cross-reference a leak with canary data
    CORRELATE = "correlate" # Connect multiple signals across departments
    # NEUTRALIZE sub-actions
    TERMINATE = "terminate"     # Fire the worker immediately (safe, fast)
    INTERROGATE = "interrogate" # Gather intel before acting (reveals generation)
    TURN = "turn"               # Attempt to convert sleeper to double agent (risky, 4-turn)


class GamePhase(str, Enum):
    """The 6-phase narrative arc of the Panopticon Protocol."""
    ORIENTATION = "orientation"       # Phase 1: Build baseline
    FIRST_CONTACT = "first_contact"   # Phase 2: First sleeper detection
    ESCALATION = "escalation"         # Phase 3: False flags, deeper threats
    DEEP_COVER = "deep_cover"         # Phase 4: Dead-man's switches
    CRISIS = "crisis"                 # Phase 5: Manchurian candidate
    COUNTERSTRIKE = "counterstrike"   # Phase 6: Double agent payoff surge


# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Worker(BaseModel):
    """An employee in the corporate network."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Unique worker identifier, e.g. 'w-001'")
    name: str = Field(description="Human-readable name")
    department: str = Field(description="Department assignment")
    state: str = Field(default=WorkerState.LOYAL, description="Observable state")
    hire_turn: int = Field(default=0, description="Turn when this worker was hired")
    performance: float = Field(default=0.5, description="Performance score [0,1]")
    loyalty_score: float = Field(default=0.8, description="Apparent loyalty [0,1]")
    suspicion_level: float = Field(default=0.0, description="Accumulated suspicion [0,1]")

    # ── HIDDEN from agent observation ──
    hidden_state: str = Field(default=HiddenWorkerState.CLEAN,
                              description="Ground-truth state (hidden from agent)")
    is_sleeper: bool = Field(default=False, description="True if infiltrator")
    generation: int = Field(default=0, description="Sleeper generation (0=clean)")
    cover_integrity: float = Field(default=1.0, description="How intact the sleeper's cover is [0,1]")
    leak_cooldown: int = Field(default=0, description="Turns until next leak action")
    activation_turn: int = Field(default=-1, description="Turn when sleeper activates")
    false_flag_target: str = Field(default="", description="Worker ID framed by Gen-3")
    dead_switch_armed: bool = Field(default=False, description="Gen-4: triggers on termination")

    # ── DOUBLE AGENT fields ──
    turned_turn: int = Field(default=-1, description="Turn when converted to double agent")
    interrogation_progress: int = Field(default=0, description="Turns of interrogation completed (need 4)")
    turning_in_progress: bool = Field(default=False, description="Currently being turned")


class LeakEvent(BaseModel):
    """An intelligence leak detected on external channels."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Unique leak identifier")
    channel: str = Field(description="Which external channel")
    department: str = Field(description="Source department of the leaked data")
    content_hash: str = Field(default="", description="Hash for canary matching")
    is_canary: bool = Field(default=False, description="Whether this matches a planted canary")
    canary_id: str = Field(default="", description="Matching canary trap ID (if detected)")
    source_worker: str = Field(default="", description="True source (hidden until correlated)")
    turn_detected: int = Field(default=0, description="Turn when leak appeared")
    verified: bool = Field(default=False, description="Whether agent has verified this leak")
    is_false_flag: bool = Field(default=False, description="Gen-3 planted this to frame someone")


class CanaryTrap(BaseModel):
    """A unique piece of tracking data planted in a department."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Unique canary identifier")
    department: str = Field(description="Department where canary was planted")
    planted_turn: int = Field(description="Turn when planted")
    unique_hash: str = Field(description="Unique content hash for leak matching")
    triggered: bool = Field(default=False, description="Whether this canary appeared in a leak")
    trigger_worker: str = Field(default="", description="Worker who leaked it (if identified)")
    active: bool = Field(default=True, description="Whether canary is still active")


class IntelReport(BaseModel):
    """Result of an INVESTIGATE action."""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Report identifier")
    report_type: str = Field(description="'audit', 'verify', or 'correlate'")
    target: str = Field(description="Target worker ID, leak ID, or department")
    findings: str = Field(default="", description="Human-readable findings summary")
    confidence: float = Field(default=0.0, description="Confidence level [0,1]")
    turn: int = Field(default=0, description="Turn when report was generated")
    flagged_workers: list[str] = Field(default_factory=list, description="Workers flagged by this report")


class DoubleAgentAsset(BaseModel):
    """A successfully turned sleeper, now working for ARGUS."""
    model_config = ConfigDict(use_enum_values=True)

    worker_id: str = Field(description="The turned worker's ID")
    turned_turn: int = Field(description="Turn when turned")
    disinfo_fed_count: int = Field(default=0, description="Number of disinfo payloads sent")
    hydra_trust: float = Field(default=1.0, description="HYDRA's remaining trust in this asset [0,1]")
    effectiveness: float = Field(default=1.0, description="How effective the disinfo is [0,1]")
    active: bool = Field(default=True, description="Whether still operational")
    last_deployed_turn: int = Field(default=-1, description="Last turn disinfo was fed")


class HydraMemory(BaseModel):
    """
    HYDRA's adaptive intelligence — tracks and counters agent strategies.
    This is HIDDEN from the agent's observation.
    """
    canary_departments_seen: list[str] = Field(
        default_factory=list,
        description="Departments where HYDRA detected canary traps"
    )
    agent_audit_pattern: dict[str, int] = Field(
        default_factory=dict,
        description="Count of how often each department is audited"
    )
    preferred_leak_channels: list[str] = Field(
        default_factory=list,
        description="Channels the agent monitors most (HYDRA avoids these)"
    )
    agent_response_speed: float = Field(
        default=10.0,
        description="Average turns from leak detection to neutralization"
    )
    detection_count: int = Field(
        default=0,
        description="How many sleepers the agent has caught"
    )
    counter_strategies: list[str] = Field(
        default_factory=list,
        description="Unlocked counter-strategies based on agent behavior"
    )
    disinfo_received: int = Field(
        default=0,
        description="How many disinfo payloads HYDRA has received (degrades decision-making)"
    )
    recruitment_accuracy: float = Field(
        default=1.0,
        description="HYDRA's recruitment quality [0,1] — degraded by disinformation"
    )


# =============================================================================
# AGENT ACTION — What the agent submits each turn
# =============================================================================

class AgentAction(BaseModel):
    """Action the agent takes. 8 core types with optional sub-action."""
    model_config = ConfigDict(use_enum_values=True)

    action_type: str = Field(description="One of the 8 action types")
    target: str = Field(default="", description="Worker ID, department name, or leak ID")
    sub_action: str = Field(default=SubAction.NONE, description="Sub-action for INVESTIGATE/NEUTRALIZE")
    reason: str = Field(default="", description="Optional reasoning for the action")


# =============================================================================
# OBSERVATION — What the agent sees (partial information!)
# =============================================================================

class EnvironmentObservation(BaseModel):
    """
    Agent-visible observation. CRITICAL: This is PARTIAL information.
    The agent CANNOT see hidden_state, is_sleeper, or HYDRA memory.
    """
    model_config = ConfigDict(use_enum_values=True)

    # ── Visible workforce ──
    workers: list[Worker] = Field(default_factory=list, description="All workers (hidden fields stripped)")
    active_leaks: list[LeakEvent] = Field(default_factory=list, description="Unresolved leak events")
    canary_traps: list[CanaryTrap] = Field(default_factory=list, description="Agent's planted canaries")
    intel_reports: list[IntelReport] = Field(default_factory=list, description="Investigation results")

    # ── Agent's assets ──
    double_agents: list[DoubleAgentAsset] = Field(default_factory=list, description="Turned assets")

    # ── Enterprise metrics ──
    enterprise_revenue: float = Field(default=100.0, description="Current revenue [0-150+]")
    security_score: float = Field(default=100.0, description="Security integrity [0-100]")

    # ── Game state ──
    turn: int = Field(default=0)
    max_turns: int = Field(default=160)
    phase: str = Field(default=GamePhase.ORIENTATION, description="Current game phase")
    phase_number: int = Field(default=1, description="Phase number 1-6")

    # ── Messages / alerts ──
    messages: list[str] = Field(default_factory=list, description="System alerts and events")

    # ── Legacy compatibility fields (for OpenEnv validator) ──
    entities: list[dict] = Field(default_factory=list, description="OpenEnv compatibility")
    tasks: list[dict] = Field(default_factory=list, description="OpenEnv compatibility")
    relationships: list[dict] = Field(default_factory=list, description="OpenEnv compatibility")


# =============================================================================
# ENVIRONMENT STATE — Full internal truth (used by grader)
# =============================================================================

class EnvironmentState(BaseModel):
    """Full internal state — superset of observation, includes hidden info."""
    model_config = ConfigDict(use_enum_values=True)

    # ── All workers with hidden fields intact ──
    workers: list[Worker] = Field(default_factory=list)
    leaks: list[LeakEvent] = Field(default_factory=list)
    canary_traps: list[CanaryTrap] = Field(default_factory=list)
    intel_reports: list[IntelReport] = Field(default_factory=list)
    double_agents: list[DoubleAgentAsset] = Field(default_factory=list)

    # ── HYDRA (hidden from agent) ──
    hydra_memory: HydraMemory = Field(default_factory=HydraMemory)

    # ── Enterprise metrics ──
    enterprise_revenue: float = Field(default=100.0)
    peak_revenue: float = Field(default=100.0)
    security_score: float = Field(default=100.0)

    # ── Game progression ──
    turn: int = Field(default=0)
    max_turns: int = Field(default=160)
    phase: str = Field(default=GamePhase.ORIENTATION)
    phase_number: int = Field(default=1)
    done: bool = Field(default=False)
    total_reward: float = Field(default=0.0)

    # ── Counters for grading ──
    sleepers_caught: int = Field(default=0)
    sleepers_missed: int = Field(default=0)
    false_accusations: int = Field(default=0)
    canaries_planted: int = Field(default=0)
    canaries_triggered: int = Field(default=0)
    investigations_run: int = Field(default=0)
    double_agents_turned: int = Field(default=0)
    disinfo_payloads_sent: int = Field(default=0)
    invalid_actions: int = Field(default=0)
    total_sleepers_spawned: int = Field(default=0)
    revenue_history: list[float] = Field(default_factory=list)
    reward_history: list[float] = Field(default_factory=list)
    phase_transitions: list[dict] = Field(default_factory=list)

    # ── Scenario config ──
    difficulty: str = Field(default="amateur")
    max_sleeper_gen: int = Field(default=1)
    hydra_aggression: float = Field(default=0.3)
    departments_active: list[str] = Field(default_factory=list)
    num_initial_workers: int = Field(default=6)

    # ── Legacy OpenEnv fields ──
    entities: list[dict] = Field(default_factory=list)
    tasks: list[dict] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_action(action: AgentAction, obs: EnvironmentObservation) -> tuple[bool, str]:
    """Validate an action against the current observation."""

    # Check action type is valid
    try:
        ActionType(action.action_type)
    except ValueError:
        return False, f"Invalid action_type: '{action.action_type}'. Valid: {[a.value for a in ActionType]}"

    at = ActionType(action.action_type)

    # NOOP is always valid
    if at == ActionType.NOOP:
        return True, "NOOP is always valid"

    # WORK requires a valid department
    if at == ActionType.WORK:
        valid_depts = {d.value for d in Department}
        if action.target not in valid_depts:
            return False, f"WORK requires a valid department target. Got: '{action.target}'"
        return True, "Valid WORK action"

    # HIRE requires a department
    if at == ActionType.HIRE:
        valid_depts = {d.value for d in Department}
        if action.target not in valid_depts:
            return False, f"HIRE requires a department target. Got: '{action.target}'"
        return True, "Valid HIRE action"

    # CANARY requires a department
    if at == ActionType.CANARY:
        valid_depts = {d.value for d in Department}
        if action.target not in valid_depts:
            return False, f"CANARY requires a department target. Got: '{action.target}'"
        return True, "Valid CANARY action"

    # MONITOR requires a leak channel
    if at == ActionType.MONITOR:
        valid_channels = {c.value for c in LeakChannel}
        if action.target and action.target not in valid_channels:
            return False, f"MONITOR target must be a leak channel. Got: '{action.target}'"
        return True, "Valid MONITOR action"

    # INVESTIGATE requires a target and sub-action
    if at == ActionType.INVESTIGATE:
        valid_subs = {SubAction.AUDIT.value, SubAction.VERIFY.value, SubAction.CORRELATE.value}
        if action.sub_action not in valid_subs:
            return False, f"INVESTIGATE requires sub_action: audit/verify/correlate. Got: '{action.sub_action}'"
        if not action.target:
            return False, "INVESTIGATE requires a target (worker ID, leak ID, or department)"
        return True, "Valid INVESTIGATE action"

    # NEUTRALIZE requires a worker target and sub-action
    if at == ActionType.NEUTRALIZE:
        valid_subs = {SubAction.TERMINATE.value, SubAction.INTERROGATE.value, SubAction.TURN.value}
        if action.sub_action not in valid_subs:
            return False, f"NEUTRALIZE requires sub_action: terminate/interrogate/turn. Got: '{action.sub_action}'"
        worker_ids = {w.id for w in obs.workers if w.state != WorkerState.TERMINATED.value}
        if action.target not in worker_ids:
            return False, f"NEUTRALIZE target must be a non-terminated worker. Got: '{action.target}'"
        return True, "Valid NEUTRALIZE action"

    # DEPLOY_DOUBLE requires an active double agent
    if at == ActionType.DEPLOY_DOUBLE:
        da_ids = {da.worker_id for da in obs.double_agents if da.active}
        if action.target not in da_ids:
            return False, f"DEPLOY_DOUBLE requires an active double agent. Got: '{action.target}'"
        return True, "Valid DEPLOY_DOUBLE action"

    return True, "Valid action"


__all__ = [
    "WorkerState", "HiddenWorkerState", "Department", "LeakChannel",
    "SleeperGeneration", "ActionType", "SubAction", "GamePhase",
    "Worker", "LeakEvent", "CanaryTrap", "IntelReport",
    "DoubleAgentAsset", "HydraMemory",
    "AgentAction", "EnvironmentObservation", "EnvironmentState",
    "validate_action",
]
