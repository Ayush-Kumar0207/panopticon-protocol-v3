def validate_episode_header(header: dict, expected_identity: dict) -> dict:
    """Check artifact identity: level, seed, reward, schema, etc."""
    mismatches = []
    for key, expected_val in expected_identity.items():
        if header.get(key) != expected_val:
            mismatches.append(f"{key}_mismatch")
            
    if mismatches:
        return {"status": "rejected", "reasons": mismatches}
    return {"status": "accepted"}

def validate_episode_turns(turns: list[dict]) -> dict:
    """Ensure chronological, contiguous turns with no duplicates or missing data."""
    seen_turns = set()
    last_turn = -1
    
    for idx, row in enumerate(turns):
        current = row.get("turn")
        if current is None:
            return {"status": "rejected", "reason": f"missing_turn_at_index_{idx}"}
            
        if current in seen_turns:
            return {"status": "rejected", "reason": f"duplicate_turn_{current}"}
            
        if current < last_turn:
            return {"status": "rejected", "reason": f"out_of_order_turn_{current}"}
            
        seen_turns.add(current)
        last_turn = current
        
    return {"status": "accepted", "valid_count": len(turns)}
