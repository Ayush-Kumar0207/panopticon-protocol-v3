"""Synthetic Stage 1 novelty scoring and episode-held-out calibration.

This module assumes upstream, versioned observation features have ALREADY been
extracted with a fixed, reviewed feature schema. It never reads real artifacts.
"""

from collections import defaultdict
from statistics import median
import math

import numpy as np

from .replay import SUPPORTED_LEVELS


def _feature_schema(features: dict) -> dict[str, str] | None:
    if not isinstance(features, dict) or not features:
        return None
    schema = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name:
            return None
        if type(value) in (int, float) and math.isfinite(value):
            schema[name] = "numeric"
        elif type(value) in (str, bool):
            schema[name] = "categorical"
        else:
            return None
    return schema



def _level_contract_error(reference_episodes, expected_level):
    """Fail closed on missing or unsupported level stratification."""
    if expected_level is None:
        if any(
            type(row) is dict
            and ("level" in row or "seed" in row)
            for row in reference_episodes
        ):
            return "missing_expected_level"
        # Legacy, level-free synthetic fixtures only.
        return None

    if (
        type(expected_level) is not str
        or expected_level not in SUPPORTED_LEVELS
    ):
        return "invalid_expected_level"

    return None


def fit_reference_scaler(expert_features: list[dict]) -> dict:
    """Fit robust numeric scaling using expert-reference features only."""
    if not expert_features:
        return {}
    schema = _feature_schema(expert_features[0])
    if schema is None or any(_feature_schema(f) != schema for f in expert_features):
        raise ValueError("expert reference has missing or inconsistent features")
    scaler = {}
    for name, kind in schema.items():
        if kind == "numeric":
            vals = [float(f[name]) for f in expert_features]
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            scaler[name] = {
                "median": float(np.median(vals)),
                "iqr": iqr if iqr > 0 else 1.0,
            }
    return scaler


def calculate_distance(query_features: dict, ref_features: dict, scaler: dict) -> float | None:
    """Return mixed numeric/categorical distance, or None if incomparable."""
    schema = _feature_schema(query_features)
    if schema is None or _feature_schema(ref_features) != schema:
        return None
    distances = []
    for name, kind in schema.items():
        if kind == "numeric":
            if name not in scaler or scaler[name]["iqr"] <= 0:
                return None
            delta = (float(query_features[name]) - float(ref_features[name])) / scaler[name]["iqr"]
            distances.append(min(5.0, abs(delta)))
        else:
            distances.append(float(query_features[name] != ref_features[name]))
    return sum(distances) / len(distances) if distances else None


def _episode_key(row: dict) -> tuple | None:
    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        return None
    if "seed" in row:
        seed = row["seed"]
        level = row.get("level")
        if type(seed) is not int or not isinstance(level, str) or not level:
            return None
        return ("seed", level, seed)
    return ("episode_id", episode_id)  # synthetic fixtures without seed


def _group_references(reference_episodes: list[dict], expected_level: str | None):
    groups = defaultdict(list)
    for row in reference_episodes:
        if not isinstance(row, dict):
            return None
        if expected_level is not None and row.get("level") != expected_level:
            continue
        key = _episode_key(row)
        if key is None or _feature_schema(row.get("features")) is None:
            return None
        groups[key].append(row)
    return groups


def score_learner_novelty(
    learner_features: dict,
    reference_episodes: list[dict],
    query_episode_id: str,
    k: int = 5,
    *,
    expected_level: str | None = None,
    query_seed: int | None = None,
) -> dict:
    """Average the closest observation from each of k distinct expert episodes."""
    schema = _feature_schema(learner_features)
    if schema is None:
        return {"status": "unavailable", "reason": "invalid_query_features"}
    if type(k) is not int or k < 1:
        return {"status": "unavailable", "reason": "invalid_k"}
    if query_seed is not None and (
        type(query_seed) is not int
        or query_seed < 0
        or type(expected_level) is not str
        or not expected_level
    ):
        return {"status": "unavailable", "reason": "invalid_query_seed"}
    if not isinstance(reference_episodes, list):
        return {"status": "unavailable", "reason": "invalid_reference"}
    level_error = _level_contract_error(
        reference_episodes, expected_level
    )
    if level_error is not None:
        return {"status": "unavailable", "reason": level_error}
    groups = _group_references(reference_episodes, expected_level)
    if groups is None:
        return {"status": "unavailable", "reason": "invalid_reference"}
    groups = {
        key: rows for key, rows in groups.items()
        if all(row["episode_id"] != query_episode_id for row in rows)
        and (
            query_seed is None
            or key != ("seed", expected_level, query_seed)
        )
    }
    if len(groups) < k:
        return {"status": "unavailable", "reason": "insufficient_reference_episodes"}
    refs = [r for rows in groups.values() for r in rows]
    if any(_feature_schema(r["features"]) != schema for r in refs):
        return {"status": "unavailable", "reason": "incomparable_features"}
    scaler = fit_reference_scaler([r["features"] for r in refs])
    per_episode = []
    for rows in groups.values():
        dists = [calculate_distance(learner_features, row["features"], scaler) for row in rows]
        if any(d is None for d in dists):
            return {"status": "unavailable", "reason": "incomparable_features"}
        per_episode.append(min(dists))
    nearest = sorted(per_episode)[:k]
    return {
        "status": "available",
        "novelty_score": float(sum(nearest) / k),
        "neighbors_used": k,
        "independent_reference_episodes": len(groups),
    }



def score_learner_episode_novelty(
    learner_turn_features: list[dict],
    reference_episodes: list[dict],
    query_episode_id: str,
    *,
    query_seed: int,
    expected_level: str,
    threshold: float | None = None,
    k: int = 5,
) -> dict:
    """Synthetic episode-median novelty; no provenance verification."""
    if not isinstance(learner_turn_features, list) or not learner_turn_features:
        return {"status": "unavailable", "reason": "invalid_learner_episode"}

    if type(query_episode_id) is not str or not query_episode_id.strip():
        return {"status": "unavailable", "reason": "invalid_query_episode"}

    if type(query_seed) is not int or query_seed < 0:
        return {"status": "unavailable", "reason": "invalid_query_seed"}

    if type(expected_level) is not str or not expected_level.strip():
        return {"status": "unavailable", "reason": "invalid_level"}

    if type(k) is not int or k != 5:
        return {"status": "unavailable", "reason": "invalid_k"}

    if threshold is not None and (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
    ):
        return {"status": "unavailable", "reason": "invalid_threshold"}

    scores = []
    independent_groups = None

    for features in learner_turn_features:
        result = score_learner_novelty(
            features,
            reference_episodes,
            query_episode_id,
            k,
            expected_level=expected_level,
            query_seed=query_seed,
        )
        if result["status"] != "available":
            return {
                "status": "unavailable",
                "reason": result["reason"],
            }

        scores.append(result["novelty_score"])
        independent_groups = result["independent_reference_episodes"]

    episode_score = float(median(scores))
    result = {
        "status": "available",
        "episode_novelty_score": episode_score,
        "unit": "episode",
        "aggregation": "median",
        "turn_scores": scores,
        "turn_count": len(scores),
        "neighbors_used": k,
        "independent_reference_episodes": independent_groups,
    }

    if threshold is not None:
        result["low_coverage"] = episode_score > threshold

    return result

def calibrate_expert_threshold(
    reference_episodes: list[dict],
    *,
    k: int = 5,
    percentile: float = 95.0,
    expected_level: str | None = None,
) -> dict:
    """Calibrate using leave-one-expert-episode/seed-out novelty scores.

    Aggregate within an episode before taking a percentile so long episodes
    do not contribute more independent calibration samples.
    """
    if not isinstance(reference_episodes, list) or not 0 < percentile < 100:
        return {"status": "unavailable", "reason": "invalid_calibration_input"}
    level_error = _level_contract_error(
        reference_episodes, expected_level
    )
    if level_error is not None:
        return {"status": "unavailable", "reason": level_error}
    groups = _group_references(reference_episodes, expected_level)
    if groups is None:
        return {"status": "unavailable", "reason": "invalid_reference"}
    if type(k) is not int or k < 1 or len(groups) < k + 1:
        return {"status": "unavailable", "reason": "insufficient_calibration_episodes"}
    episode_scores = []
    for held_out_key, held_out_rows in groups.items():
        training_refs = [r for group_key, rows in groups.items() if group_key != held_out_key for r in rows]
        turn_scores = []
        for held_out in held_out_rows:
            result = score_learner_novelty(
                held_out["features"], training_refs, held_out["episode_id"], k,
                expected_level=expected_level,
                query_seed=(
                    held_out_key[2]
                    if held_out_key[0] == "seed"
                    and expected_level is not None
                    else None
                ),
            )
            if result["status"] != "available":
                return {"status": "unavailable", "reason": f"calibration_{result['reason']}"}
            turn_scores.append(result["novelty_score"])
        episode_scores.append(float(median(turn_scores)))
    return {
        "status": "available",
        "threshold": float(np.percentile(episode_scores, percentile)),
        "threshold_unit": "episode_median",
        "threshold_comparator": ">",
        "percentile": float(percentile),
        "calibration_episodes": len(episode_scores),
        "held_out_episode_scores": episode_scores,
    }
