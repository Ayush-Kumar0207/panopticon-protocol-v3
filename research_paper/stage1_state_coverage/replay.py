from copy import deepcopy

# Import upstream root modules as instructed by the guide
from models import validate_action
from security_policy import choose_security_first_action, new_security_expert_state

# Import your new canonical utility
from .canonical import canonicalize_action

class OracleUnavailable(Exception):
    """Required observable history or validity evidence is missing/ambiguous."""

def reconstruct_expert_state(prior_turns: list[dict]) -> dict:
    state = new_security_expert_state()
    for row in prior_turns:
        if "executed_action" not in row or "executed_semantic_valid" not in row:
            raise OracleUnavailable("missing executed action or semantic-validity evidence")
        
        info = row.get("info")
        if not isinstance(info, dict) or "valid" not in info:
            raise OracleUnavailable("missing recorded environment validity")
            
        semantic_valid, environment_valid = row["executed_semantic_valid"], info["valid"]
        if type(semantic_valid) is not bool or type(environment_valid) is not bool:
            raise OracleUnavailable("non-boolean validity evidence")
            
        if not (semantic_valid and environment_valid):
            continue
            
        action = canonicalize_action(row["executed_action"])
        kind, target, sub = action
        
        if kind == "investigate" and sub == "audit":
            state["audit_idx"] += 1
        elif kind == "monitor":
            state["monitor_idx"] += 1
        elif kind == "canary":
            state["canaried_departments"].add(target)
        elif kind == "neutralize" and sub == "turn":
            state["turned_sleeper"] = True
            
    return state
