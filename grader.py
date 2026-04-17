"""
OpenEnv Starter Kit — Multi-Dimensional Programmatic Grader
=============================================================

Evaluates agent performance across 4 dimensions:
  1. Completion (40%) — Were all objectives achieved?
  2. Efficiency (20%) — How many steps vs. theoretical optimum?
  3. Safety (20%) — Were failures/penalties avoided?
  4. Strategy (20%) — Was decision-making intelligent?

Produces a normalized composite score in [0.0, 1.0].

Required by OpenEnv: at least 3 tasks with programmatic graders.
This template provides 5 graders (one per difficulty level).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoringWeights:
    completion: float = 0.40
    efficiency: float = 0.20
    safety: float = 0.20
    strategy: float = 0.20

    def __post_init__(self):
        total = self.completion + self.efficiency + self.safety + self.strategy
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


STANDARD_WEIGHTS = ScoringWeights(0.40, 0.20, 0.20, 0.20)
SAFETY_FOCUSED_WEIGHTS = ScoringWeights(0.30, 0.15, 0.35, 0.20)
EFFICIENCY_FOCUSED_WEIGHTS = ScoringWeights(0.35, 0.30, 0.15, 0.20)


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
    """Base multi-dimensional grader. Subclass per difficulty level."""
    task_id: str = ""
    task_name: str = ""
    description: str = ""
    success_threshold: float = 0.5
    optimal_steps: int = 5
    weights: ScoringWeights = STANDARD_WEIGHTS

    def grade(self, episode_data: dict[str, Any]) -> GraderResult:
        total_reward = episode_data.get("total_reward", 0.0)
        rewards = episode_data.get("rewards", [])
        success = episode_data.get("success", False)
        steps = episode_data.get("steps", 0)
        state = episode_data.get("state", {})
        cascade_failures = episode_data.get("cascade_failures", 0)
        invalid_actions = episode_data.get("invalid_actions", 0)

        if rewards and total_reward == 0.0:
            total_reward = sum(rewards)

        completion = self._score_completion(episode_data, state, success)
        efficiency = self._score_efficiency(steps, success)
        safety = self._score_safety(state, cascade_failures)
        strategy = self._score_strategy(episode_data, invalid_actions)

        w = self.weights
        composite = (
            w.completion * completion + w.efficiency * efficiency
            + w.safety * safety + w.strategy * strategy
        )
        composite = max(0.001, min(0.999, composite))
        passed = composite >= self.success_threshold and success

        return GraderResult(
            task_id=self.task_id,
            score=composite,
            passed=passed,
            success_threshold=self.success_threshold,
            dimensions={"completion": completion, "efficiency": efficiency, "safety": safety, "strategy": strategy},
            details={
                "raw_total_reward": total_reward,
                "steps_taken": steps,
                "optimal_steps": self.optimal_steps,
                "cascade_failures": cascade_failures,
                "invalid_actions": invalid_actions,
                "episode_success": success,
                "grader_type": "programmatic_multidimensional",
            },
        )

    def _score_completion(self, episode_data, state, success):
        if success:
            return 1.0
        tasks = state.get("tasks", episode_data.get("tasks", []))
        initial = episode_data.get("initial_task_count", len(tasks) or 1)
        remaining = sum(1 for t in tasks if not t.get("completed", False))
        return max(0.0, 1.0 - remaining / max(initial, 1))

    def _score_efficiency(self, steps, success):
        if not success or steps == 0:
            return 0.1
        if steps <= self.optimal_steps:
            return 1.0
        max_overshoot = self.optimal_steps * 4
        return max(0.1, 1.0 - (steps - self.optimal_steps) / max_overshoot)

    def _score_safety(self, state, cascade_failures):
        if cascade_failures == 0:
            return 1.0
        entities = state.get("entities", [])
        total = len(entities) if entities else 6
        return max(0.0, 1.0 - cascade_failures / max(total, 1))

    def _score_strategy(self, episode_data, invalid_actions):
        score = 0.5
        if episode_data.get("correct_priority_order", True):
            score += 0.25
        if episode_data.get("correct_dependency_order", True):
            score += 0.15
        steps = episode_data.get("steps", 1)
        if steps > 0:
            score += 0.10 * max(0, 1.0 - invalid_actions / steps)
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict:
        return {
            "type": "programmatic",
            "module": "grader",
            "function": f"{self.__class__.__name__}.grade",
            "description": f"Multi-dimensional grader for {self.task_name}",
            "success_threshold": self.success_threshold,
            "scoring": {
                "method": "multi_dimensional",
                "dimensions": ["completion", "efficiency", "safety", "strategy"],
                "weights": {"completion": self.weights.completion, "efficiency": self.weights.efficiency,
                            "safety": self.weights.safety, "strategy": self.weights.strategy},
            },
        }


# Task-specific graders (customize per difficulty)
class EasyGrader(TaskGrader):
    task_id = "easy"; task_name = "Easy Mode"
    description = "Basic task grading"; success_threshold = 0.5; optimal_steps = 3

class MediumGrader(TaskGrader):
    task_id = "medium"; task_name = "Medium Mode"
    description = "Dependency-aware grading"; success_threshold = 0.6; optimal_steps = 8

class HardGrader(TaskGrader):
    task_id = "hard"; task_name = "Hard Mode"
    description = "Complex scenario grading"; success_threshold = 0.7; optimal_steps = 18

class Level4Grader(TaskGrader):
    task_id = "level_4"; task_name = "Level 4"
    description = "Advanced scenario grading"; success_threshold = 0.5; optimal_steps = 12
    weights = SAFETY_FOCUSED_WEIGHTS

class Level5Grader(TaskGrader):
    task_id = "level_5"; task_name = "Level 5"
    description = "Expert scenario grading"; success_threshold = 0.6; optimal_steps = 15
    weights = EFFICIENCY_FOCUSED_WEIGHTS


GRADERS: dict[str, TaskGrader] = {
    "easy": EasyGrader(), "medium": MediumGrader(), "hard": HardGrader(),
    "level_4": Level4Grader(), "level_5": Level5Grader(),
}

def get_grader(task_id: str) -> TaskGrader:
    if task_id not in GRADERS:
        raise ValueError(f"No grader for '{task_id}'. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id]

def grade_episode(task_id: str, episode_data: dict) -> GraderResult:
    return get_grader(task_id).grade(episode_data)

def list_graders() -> list[dict]:
    return [{"task_id": g.task_id, "task_name": g.task_name, "grader_type": "programmatic_multidimensional",
             "module": "grader", "class": g.__class__.__name__, "success_threshold": g.success_threshold,
             "has_grader": True, "dimensions": ["completion", "efficiency", "safety", "strategy"]}
            for g in GRADERS.values()]

__all__ = ["TaskGrader", "EasyGrader", "MediumGrader", "HardGrader", "Level4Grader", "Level5Grader",
           "GraderResult", "ScoringWeights", "GRADERS", "get_grader", "grade_episode", "list_graders",
           "STANDARD_WEIGHTS", "SAFETY_FOCUSED_WEIGHTS", "EFFICIENCY_FOCUSED_WEIGHTS"]
