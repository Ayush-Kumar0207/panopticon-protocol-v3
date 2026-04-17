"""Task definition: Hard Mode. TODO: Update description for your problem."""
from grader import HardGrader

_grader = HardGrader()

HARD_TASK = {
    "id": "hard",
    "name": "Hard Mode",
    "description": "TODO: Describe this difficulty level for your problem.",
    "max_turns": 100,
    "difficulty": 3,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_tasks_completed": True,
        "no_catastrophic_failures": True,
    },
}
