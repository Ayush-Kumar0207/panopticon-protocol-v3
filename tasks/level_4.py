"""Task definition: Level 4. TODO: Update description for your problem."""
from grader import Level4Grader

_grader = Level4Grader()

LEVEL_4_TASK = {
    "id": "level_4",
    "name": "Level 4",
    "description": "TODO: Describe this difficulty level for your problem.",
    "max_turns": 60,
    "difficulty": 4,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_tasks_completed": True,
        "no_catastrophic_failures": True,
    },
}
