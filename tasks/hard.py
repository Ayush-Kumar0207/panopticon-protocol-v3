"""Task definition: Spy Network — Gen-1/2/3 sleepers with false flags."""
from grader import SpyGrader

_grader = SpyGrader()

HARD_TASK = {
    "id": "hard",
    "name": "Spy Network",
    "description": (
        "HYDRA deploys three generations of sleepers, culminating in a Gen-3 spy "
        "who plants FALSE EVIDENCE framing innocent workers. Requires verification "
        "to avoid false accusations. Trust nothing without canary confirmation. "
        "120 turns. 8 workers across 5 departments."
    ),
    "max_turns": 120,
    "difficulty": 3,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "majority_sleepers_caught": True,
        "no_innocent_terminated": True,
        "revenue_above_40": True,
    },
}
