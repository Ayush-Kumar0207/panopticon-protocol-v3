"""Task registry — 5 Panopticon Protocol difficulty tiers with graders."""
from __future__ import annotations
from tasks.easy import EASY_TASK
from tasks.medium import MEDIUM_TASK
from tasks.hard import HARD_TASK
from tasks.level_4 import LEVEL_4_TASK
from tasks.level_5 import LEVEL_5_TASK

TASK_REGISTRY: dict[str, dict] = {
    "easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK,
    "level_4": LEVEL_4_TASK, "level_5": LEVEL_5_TASK,
}

def get_task(task_id: str) -> dict:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Task '{task_id}' not found. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]

def list_tasks() -> list[dict]:
    return list(TASK_REGISTRY.values())

def list_tasks_with_graders() -> list[dict]:
    return [t for t in TASK_REGISTRY.values() if t.get("has_grader", False)]

__all__ = ["TASK_REGISTRY", "get_task", "list_tasks", "list_tasks_with_graders"]
