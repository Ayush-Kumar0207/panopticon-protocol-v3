"""Task definition: Level 5. TODO: Update description for your problem."""
from grader import Level5Grader

_grader = Level5Grader()

LEVEL_5_TASK = {
    "id": "level_5",
    "name": "Level 5",
    "description": "TODO: Describe this difficulty level for your problem.",
    "max_turns": 80,
    "difficulty": 5,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_tasks_completed": True,
        "no_catastrophic_failures": True,
    },
}
