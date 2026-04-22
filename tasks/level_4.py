"""Task definition: Terror Cell — Gen-1 through Gen-4 with dead-man's switches."""
from grader import CellGrader

_grader = CellGrader()

LEVEL_4_TASK = {
    "id": "level_4",
    "name": "Terror Cell",
    "description": (
        "HYDRA escalates to Gen-4 operatives equipped with DEAD MAN'S SWITCHES — "
        "terminating without interrogation first causes massive revenue damage. "
        "The agent must INTERROGATE before TERMINATING to detect armed switches. "
        "Double agent TURNING becomes strategically critical. "
        "150 turns. 9 workers across 6 departments."
    ),
    "max_turns": 150,
    "difficulty": 4,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "no_dead_switch_triggered": True,
        "double_agent_turned": True,
        "revenue_above_40": True,
    },
}
