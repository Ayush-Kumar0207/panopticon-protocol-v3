def canonicalize_action(raw_action: dict) -> tuple[str, str, str]:
    """Convert a raw action dictionary into a canonical (type, target, sub_action) tuple."""
    if not isinstance(raw_action, dict):
        return ("noop", "", "none")
    action_type = str(raw_action.get("action_type", "noop"))
    target = str(raw_action.get("target", ""))
    sub_action = raw_action.get("sub_action")
    
    # Guide specifies omitted optional sub-actions become "none"
    if not sub_action:
        sub_action = "none"
        
    return (action_type, target, str(sub_action))
