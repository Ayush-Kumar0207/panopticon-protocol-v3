"""Task definition: Professional Threat — Gen-1 + Gen-2 sleepers."""
from grader import ProfessionalGrader

_grader = ProfessionalGrader()

MEDIUM_TASK = {
    "id": "medium",
    "name": "Professional Threat",
    "description": (
        "HYDRA deploys a Gen-1 amateur followed by a Gen-2 professional who avoids "
        "canary departments and monitored channels. Requires signal correlation and "
        "multi-channel monitoring. 90 turns. 7 workers across 4 departments."
    ),
    "max_turns": 90,
    "difficulty": 2,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "both_sleepers_neutralized": True,
        "revenue_above_50": True,
        "max_1_false_accusation": True,
    },
}
