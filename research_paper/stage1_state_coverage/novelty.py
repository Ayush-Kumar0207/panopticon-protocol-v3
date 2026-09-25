import numpy as np

def fit_reference_scaler(expert_features: list[dict]) -> dict:
    """Fit median and IQR on the reference set for numeric standardization."""
    scaler = {}
    if not expert_features:
        return scaler
        
    keys = expert_features[0].keys()
    for key in keys:
        vals = [f[key] for f in expert_features if isinstance(f.get(key), (int, float)) and not isinstance(f.get(key), bool)]
        if vals:
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            # Guide specifies robust fallback when IQR == 0
            if iqr == 0.0:
                iqr = 1.0 
                
            scaler[key] = {
                "median": float(np.median(vals)),
                "iqr": iqr
            }
    return scaler

def calculate_distance(query_features: dict, ref_features: dict, scaler: dict) -> float:
    """Mixed-type Gower-style average distance with clipped standardized numeric differences."""
    distances = []
    for key, val in query_features.items():
        ref_val = ref_features.get(key)
        if ref_val is None:
            continue
            
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            median = scaler.get(key, {}).get("median", 0.0)
            iqr = scaler.get(key, {}).get("iqr", 1.0)
            
            std_diff = (val - ref_val) / iqr
            clipped = max(-5.0, min(5.0, std_diff))
            distances.append(abs(clipped))
        else:
            # Categorical: 0 if equal, 1 if unequal
            distances.append(0.0 if val == ref_val else 1.0)
            
    return sum(distances) / len(distances) if distances else 0.0

def score_learner_novelty(learner_features: dict, reference_episodes: list[dict], query_episode_id: str, k: int = 5) -> dict:
    """Score novelty against k nearest neighbors, excluding the query's own episode."""
    # Filter out the query episode to ensure no self-neighbors
    eligible_refs = [r for r in reference_episodes if r.get("episode_id") != query_episode_id]
    
    # Require at least k eligible neighbors
    if len(eligible_refs) < k:
        return {"status": "unavailable", "reason": "insufficient_reference_episodes"}
        
    scaler = fit_reference_scaler([r["features"] for r in eligible_refs])
    
    distances = []
    for ref in eligible_refs:
        dist = calculate_distance(learner_features, ref["features"], scaler)
        distances.append(dist)
        
    distances.sort()
    nearest = distances[:k]
    score = sum(nearest) / k
    
    return {
        "status": "available",
        "novelty_score": score,
        "neighbors_used": k
    }
