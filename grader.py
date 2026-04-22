"""
The Panopticon Protocol v3 — Multi-Dimensional Grader
=======================================================

Evaluates agent performance across 5 espionage-specific dimensions:
  1. Security Outcome (30%) — Sleepers caught, false accusations avoided
  2. Enterprise Revenue (25%) — Revenue maintained above threshold
  3. Intelligence Craft (20%) — Canary usage, investigation quality, DA turns
  4. Adaptability (15%) — Strategy diversity, counter-HYDRA effectiveness
  5. Efficiency (10%) — Action economy, invalid action rate

Required by OpenEnv: ≥3 tasks with programmatic graders.
This implementation provides 5 (one per difficulty tier).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoringWeights:
    security: float = 0.30
    revenue: float = 0.25
    intelligence: float = 0.20
    adaptability: float = 0.15
    efficiency: float = 0.10

    def __post_init__(self):
        total = self.security + self.revenue + self.intelligence + self.adaptability + self.efficiency
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


STANDARD_WEIGHTS = ScoringWeights(0.30, 0.25, 0.20, 0.15, 0.10)
SECURITY_HEAVY = ScoringWeights(0.40, 0.20, 0.15, 0.15, 0.10)
INTEL_HEAVY = ScoringWeights(0.25, 0.20, 0.30, 0.15, 0.10)
BALANCED = ScoringWeights(0.25, 0.25, 0.20, 0.20, 0.10)
ENDGAME = ScoringWeights(0.25, 0.25, 0.20, 0.20, 0.10)


@dataclass
class GraderResult:
    task_id: str
    score: float
    passed: bool
    success_threshold: float
    dimensions: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "score": round(self.score, 4),
            "passed": self.passed,
            "success_threshold": self.success_threshold,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "details": self.details,
        }


class TaskGrader:
    """Base multi-dimensional grader for the Panopticon Protocol."""
    task_id: str = ""
    task_name: str = ""
    description: str = ""
    success_threshold: float = 0.5
    optimal_steps: int = 30
    weights: ScoringWeights = STANDARD_WEIGHTS
    expected_sleepers: int = 1

    def grade(self, episode_data: dict[str, Any]) -> GraderResult:
        state = episode_data.get("state", {})
        steps = episode_data.get("steps", 0)
        total_reward = episode_data.get("total_reward", 0.0)
        rewards = episode_data.get("rewards", [])
        success = episode_data.get("success", False)

        if rewards and total_reward == 0.0:
            total_reward = sum(rewards)

        # Extract Panopticon-specific metrics
        sleepers_caught = state.get("sleepers_caught", 0)
        sleepers_missed = state.get("sleepers_missed", 0)
        false_accusations = state.get("false_accusations", 0)
        total_spawned = state.get("total_sleepers_spawned", max(1, self.expected_sleepers))
        canaries_planted = state.get("canaries_planted", 0)
        canaries_triggered = state.get("canaries_triggered", 0)
        investigations_run = state.get("investigations_run", 0)
        double_agents_turned = state.get("double_agents_turned", 0)
        disinfo_payloads = state.get("disinfo_payloads_sent", 0)
        invalid_actions = state.get("invalid_actions", 0)
        enterprise_revenue = state.get("enterprise_revenue", 100.0)
        security_score = state.get("security_score", 100.0)
        revenue_history = state.get("revenue_history", [100.0])
        phase_number = state.get("phase_number", 1)

        # ── Score each dimension ──
        sec = self._score_security(sleepers_caught, total_spawned, false_accusations, security_score)
        rev = self._score_revenue(enterprise_revenue, revenue_history)
        intel = self._score_intelligence(canaries_planted, canaries_triggered,
                                         investigations_run, double_agents_turned,
                                         disinfo_payloads, total_spawned)
        adapt = self._score_adaptability(state, phase_number)
        eff = self._score_efficiency(steps, invalid_actions)

        # ── Weighted composite ──
        w = self.weights
        composite = (
            w.security * sec + w.revenue * rev +
            w.intelligence * intel + w.adaptability * adapt +
            w.efficiency * eff
        )
        composite = max(0.001, min(0.999, composite))

        # Determine pass — needs composite AND security above threshold
        passed = composite >= self.success_threshold and security_score > 20

        return GraderResult(
            task_id=self.task_id,
            score=composite,
            passed=passed,
            success_threshold=self.success_threshold,
            dimensions={
                "security": sec, "revenue": rev,
                "intelligence": intel, "adaptability": adapt,
                "efficiency": eff,
            },
            details={
                "raw_total_reward": total_reward,
                "steps_taken": steps,
                "sleepers_caught": sleepers_caught,
                "total_sleepers_spawned": total_spawned,
                "false_accusations": false_accusations,
                "canaries_planted": canaries_planted,
                "double_agents_turned": double_agents_turned,
                "disinfo_payloads_sent": disinfo_payloads,
                "final_revenue": enterprise_revenue,
                "final_security": security_score,
                "phase_reached": phase_number,
                "invalid_actions": invalid_actions,
                "grader_type": "programmatic_panopticon_v3",
            },
        )

    def _score_security(self, caught: int, total: int, false_acc: int, sec_score: float) -> float:
        """How effectively the agent neutralized threats."""
        if total == 0:
            return 0.8  # No threats = decent baseline

        catch_rate = caught / max(total, 1)
        false_penalty = min(0.4, false_acc * 0.15)  # Each false accusation hurts
        sec_health = sec_score / 100.0

        score = (0.5 * catch_rate) + (0.3 * sec_health) - false_penalty + 0.2
        return max(0.0, min(1.0, score))

    def _score_revenue(self, final_revenue: float, history: list[float]) -> float:
        """How well the agent maintained enterprise productivity."""
        if not history:
            return 0.5

        # Revenue above 80 = excellent, 60 = good, below 40 = failing
        rev_score = min(1.0, final_revenue / 100.0)

        # Bonus for revenue stability (low variance)
        if len(history) > 5:
            import numpy as np
            arr = np.array(history[-20:])
            stability = 1.0 - min(1.0, np.std(arr) / 30.0)
            rev_score = 0.7 * rev_score + 0.3 * stability

        # Bonus for exceeding starting revenue (surge)
        if final_revenue > 100.0:
            rev_score = min(1.0, rev_score + 0.1)

        return max(0.0, min(1.0, rev_score))

    def _score_intelligence(self, canaries: int, triggered: int,
                            investigations: int, da_turned: int,
                            disinfo: int, total_sleepers: int) -> float:
        """How sophisticated the agent's intelligence operations were."""
        score = 0.0

        # Canary craft: planted canaries that triggered
        if canaries > 0:
            score += 0.2
            if triggered > 0:
                score += 0.2 * min(1.0, triggered / max(canaries, 1))

        # Investigation depth
        if investigations > 0:
            score += 0.15 * min(1.0, investigations / max(total_sleepers * 3, 1))

        # Double agent mastery — the pinnacle of intelligence craft
        if da_turned > 0:
            score += 0.25
            if disinfo > 0:
                score += 0.15 * min(1.0, disinfo / 3.0)

        # Minimum score if agent at least tried
        if canaries > 0 or investigations > 0:
            score = max(0.15, score)

        return max(0.0, min(1.0, score))

    def _score_adaptability(self, state: dict, phase_reached: int) -> float:
        """How diverse and adaptive the agent's strategies were."""
        score = 0.0

        # Reaching higher phases = better
        score += 0.2 * min(1.0, phase_reached / 4.0)

        # Using diverse actions
        # (Bonus for reaching Phase 6 and using double agents)
        da_turned = state.get("double_agents_turned", 0)
        if da_turned > 0:
            score += 0.3

        disinfo = state.get("disinfo_payloads_sent", 0)
        if disinfo > 0:
            score += 0.2

        # Penalty for over-reliance on termination
        caught = state.get("sleepers_caught", 0)
        false_acc = state.get("false_accusations", 0)
        if caught > 0 and false_acc == 0:
            score += 0.15  # Clean operations

        # Base score for not doing nothing
        if state.get("canaries_planted", 0) > 0 or state.get("investigations_run", 0) > 0:
            score = max(0.2, score)

        return max(0.0, min(1.0, score))

    def _score_efficiency(self, steps: int, invalid_actions: int) -> float:
        """Action economy score."""
        if steps == 0:
            return 0.1

        # Invalid action rate
        invalid_rate = invalid_actions / max(steps, 1)
        eff_score = max(0.1, 1.0 - invalid_rate * 3.0)

        # Step efficiency
        if steps <= self.optimal_steps:
            eff_score = min(1.0, eff_score + 0.1)
        elif steps > self.optimal_steps * 2:
            eff_score *= 0.8

        return max(0.0, min(1.0, eff_score))

    def to_dict(self) -> dict:
        return {
            "type": "programmatic",
            "module": "grader",
            "function": f"{self.__class__.__name__}.grade",
            "description": f"Panopticon v3 grader for {self.task_name}",
            "success_threshold": self.success_threshold,
            "scoring": {
                "method": "multi_dimensional_espionage",
                "dimensions": ["security", "revenue", "intelligence", "adaptability", "efficiency"],
                "weights": {
                    "security": self.weights.security, "revenue": self.weights.revenue,
                    "intelligence": self.weights.intelligence,
                    "adaptability": self.weights.adaptability,
                    "efficiency": self.weights.efficiency,
                },
            },
        }


# =============================================================================
# TASK-SPECIFIC GRADERS
# =============================================================================

class AmateurGrader(TaskGrader):
    task_id = "easy"
    task_name = "Amateur Threat"
    description = "Single Gen-1 sleeper. Basic canary detection."
    success_threshold = 0.4
    optimal_steps = 20
    weights = STANDARD_WEIGHTS
    expected_sleepers = 1


class ProfessionalGrader(TaskGrader):
    task_id = "medium"
    task_name = "Professional Threat"
    description = "Gen-1 + Gen-2 sleepers. Channel-aware adversary."
    success_threshold = 0.45
    optimal_steps = 35
    weights = STANDARD_WEIGHTS
    expected_sleepers = 2


class SpyGrader(TaskGrader):
    task_id = "hard"
    task_name = "Spy Network"
    description = "Gen-1/2/3 sleepers. False flags active."
    success_threshold = 0.5
    optimal_steps = 50
    weights = INTEL_HEAVY
    expected_sleepers = 3


class CellGrader(TaskGrader):
    task_id = "level_4"
    task_name = "Terror Cell"
    description = "Gen-1-4 sleepers. Dead-man's switches."
    success_threshold = 0.5
    optimal_steps = 65
    weights = SECURITY_HEAVY
    expected_sleepers = 4


class ManchurianGrader(TaskGrader):
    task_id = "level_5"
    task_name = "Manchurian Protocol"
    description = "Full 5-gen gauntlet. Counterstrike phase."
    success_threshold = 0.5
    optimal_steps = 80
    weights = ENDGAME
    expected_sleepers = 5


# =============================================================================
# REGISTRY
# =============================================================================

GRADERS: dict[str, TaskGrader] = {
    "easy": AmateurGrader(),
    "medium": ProfessionalGrader(),
    "hard": SpyGrader(),
    "level_4": CellGrader(),
    "level_5": ManchurianGrader(),
}


def get_grader(task_id: str) -> TaskGrader:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for '{task_id}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id]


def grade_episode(task_id: str, episode_data: dict) -> GraderResult:
    return get_grader(task_id).grade(episode_data)


def list_graders() -> list[dict]:
    return [
        {
            "task_id": g.task_id, "task_name": g.task_name,
            "grader_type": "programmatic_panopticon_v3",
            "module": "grader", "class": g.__class__.__name__,
            "success_threshold": g.success_threshold,
            "has_grader": True,
            "dimensions": ["security", "revenue", "intelligence", "adaptability", "efficiency"],
        }
        for g in GRADERS.values()
    ]


__all__ = [
    "TaskGrader", "AmateurGrader", "ProfessionalGrader", "SpyGrader",
    "CellGrader", "ManchurianGrader",
    "GraderResult", "ScoringWeights",
    "GRADERS", "get_grader", "grade_episode", "list_graders",
    "STANDARD_WEIGHTS", "SECURITY_HEAVY", "INTEL_HEAVY", "BALANCED", "ENDGAME",
]
