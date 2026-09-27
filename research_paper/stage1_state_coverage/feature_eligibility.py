
"""Conservative synthetic-only feature-parity screen.

This screen does not authenticate training provenance, approve a final
feature allowlist, or authorize real-artifact analysis.
"""

from ._synthetic_features import (
    CANDIDATE_FEATURE_VERSION,
    TRUNCATION_MARKER,
    candidate_header,
    candidate_triggered_canaries,
    candidate_workforce_summary,
    candidate_learner_features,
    candidate_text_leak_asset_features,
    candidate_learner_leak_asset_features,
)


def screen_synthetic_feature_parity(
    training_text,
    observation_before,
    *,
    synthetic,
    expected_feature_version,
):
    """Require complete parity for the currently tested candidate set."""

    if synthetic is not True:
        return {
            "status": "unavailable",
            "reason": "real_requires_integrated_gate",
        }

    if expected_feature_version != CANDIDATE_FEATURE_VERSION:
        return {
            "status": "unavailable",
            "reason": "feature_version_mismatch",
        }

    if type(training_text) is not str:
        return {
            "status": "unavailable",
            "reason": "invalid_training_text",
        }

    if TRUNCATION_MARKER in training_text:
        return {
            "status": "unavailable",
            "reason": "token_truncated",
        }

    learner = candidate_learner_features(observation_before)
    learner_assets = candidate_learner_leak_asset_features(
        observation_before
    )

    if (
        learner["status"] != "available"
        or learner_assets["status"] != "available"
    ):
        return {
            "status": "unavailable",
            "reason": "missing_or_invalid_learner_features",
        }

    text_sections = {
        "header": candidate_header(training_text),
        "workforce": candidate_workforce_summary(training_text),
        "leak_assets": candidate_text_leak_asset_features(
            training_text
        ),
        "triggered_canaries": candidate_triggered_canaries(
            training_text
        ),
    }

    for section, result in text_sections.items():
        if result["status"] != "available":
            return {
                "status": "unavailable",
                "reason": result["reason"],
                "section": section,
            }

        if result.get("unavailable_fields"):
            return {
                "status": "unavailable",
                "reason": "partial_training_features",
                "section": section,
                "fields": result["unavailable_fields"],
            }

    expected = {
        "header": learner["header"],
        "workforce": learner["workforce"],
        "leak_assets": learner_assets["features"],
        "triggered_canaries": learner["triggered_canaries"],
    }

    actual = {
        "header": text_sections["header"]["features"],
        "workforce": text_sections["workforce"]["features"],
        "leak_assets": text_sections["leak_assets"]["features"],
        "triggered_canaries": (
            text_sections["triggered_canaries"]["triggered_count"]
        ),
    }

    for section in expected:
        if (
            type(actual[section]) is not type(expected[section])
            or actual[section] != expected[section]
        ):
            return {
                "status": "unavailable",
                "reason": "feature_parity_mismatch",
                "section": section,
            }

    features = {
        **actual["header"],
        **actual["workforce"],
        **actual["leak_assets"],
        "triggered_canary_count": actual["triggered_canaries"],
    }

    return {
        "status": "unverified",
        "reason": "synthetic_feature_parity_only",
        "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
        "features": features,
    }
