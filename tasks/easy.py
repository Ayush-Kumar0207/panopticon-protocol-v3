"""Task definition: Amateur Threat — Gen-1 sleeper detection."""
from grader import AmateurGrader

_grader = AmateurGrader()

EASY_TASK = {
    "id": "easy",
    "name": "Amateur Threat",
    "description": (
        "A single Gen-1 sleeper has infiltrated the enterprise. "
        "The sleeper is unsophisticated — leaking canary data through obvious channels. "
        "Learn the basics: plant canaries, monitor channels, identify and neutralize the threat. "
        "60 turns. 6 workers across 3 departments."
    ),
    "max_turns": 60,
    "difficulty": 1,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "sleeper_neutralized": True,
        "revenue_above_50": True,
        "no_false_accusations": True,
    },
}
