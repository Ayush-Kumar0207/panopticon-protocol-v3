"""Task definition: Manchurian Protocol — Full 5-gen gauntlet with Counterstrike."""
from grader import ManchurianGrader

_grader = ManchurianGrader()

LEVEL_5_TASK = {
    "id": "level_5",
    "name": "Manchurian Protocol",
    "description": (
        "The ultimate challenge. All 5 sleeper generations deploy in sequence, "
        "culminating in a MANCHURIAN CANDIDATE — a top-performing worker who is "
        "secretly a Gen-5 sleeper targeting the executive department. "
        "To achieve maximum score, the agent must: "
        "(1) Survive the triple-V crisis pattern, "
        "(2) TURN at least one sleeper into a double agent, "
        "(3) Feed disinformation back to HYDRA, "
        "(4) Trigger the COUNTERSTRIKE phase where reward SURGES past initial peak. "
        "160 turns. 10 workers across 6 departments. Full HYDRA adaptive memory."
    ),
    "max_turns": 160,
    "difficulty": 5,
    "has_grader": True,
    "grader": _grader.to_dict(),
    "grader_instance": _grader,
    "success_criteria": {
        "counterstrike_reached": True,
        "double_agent_deployed": True,
        "revenue_exceeds_initial": True,
        "final_security_above_50": True,
    },
}
