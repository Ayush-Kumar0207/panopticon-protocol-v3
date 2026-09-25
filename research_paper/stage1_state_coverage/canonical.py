"""Canonical action representation for Stage 1 analysis."""

from enum import Enum


def canonicalize_action(raw_action: dict) -> tuple[str, str, str]:
    """Return (action_type, target, sub_action), normalizing enum values."""

    if not isinstance(raw_action, dict):
        raise ValueError("Action must be a dictionary")

    action_type = raw_action.get("action_type")
    target = raw_action.get("target", "")
    sub_action = raw_action.get("sub_action", "none")

    if isinstance(action_type, Enum):
        action_type = action_type.value

    if isinstance(sub_action, Enum):
        sub_action = sub_action.value

    if sub_action is None or sub_action == "":
        sub_action = "none"

    if not isinstance(action_type, str) or not action_type:
        raise ValueError("Missing or invalid action_type")

    if not isinstance(target, str):
        raise ValueError("Invalid action target")

    if not isinstance(sub_action, str):
        raise ValueError("Invalid sub_action")

    return action_type, target, sub_action
