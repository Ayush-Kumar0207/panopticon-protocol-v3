"""Task definition: Medium Mode. TODO: Update description for your problem."""
from grader import MediumGrader

_grader = MediumGrader()

MEDIUM_TASK = {
    "id": "medium",
    "name": "Medium Mode",
    "description": "TODO: Describe this difficulty level for your problem.",
    "max_turns": 50,
    "difficulty": 2,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "all_tasks_completed": True,
        "no_catastrophic_failures": True,
    },
}
