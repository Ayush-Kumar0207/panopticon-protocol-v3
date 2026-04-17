"""Task definition: Easy Mode. TODO: Update description for your problem."""
from grader import EasyGrader

_grader = EasyGrader()

EASY_TASK = {
    "id": "easy",
    "name": "Easy Mode",
    "description": "TODO: Describe this difficulty level for your problem.",
    "max_turns": 30,
    "difficulty": 1,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_tasks_completed": True,
        "no_catastrophic_failures": True,
    },
}
