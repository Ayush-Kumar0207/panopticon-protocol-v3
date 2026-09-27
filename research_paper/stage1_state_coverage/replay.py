
"""Observable learner-history replay for the Stage 1 expert oracle."""

from copy import deepcopy
import re

from models import ActionType, EnvironmentObservation, validate_action
from security_policy import (
    choose_security_first_action,
    new_security_expert_state,
)

from .canonical import canonicalize_action
from .validation import validate_episode_header, validate_episode_turns


SUPPORTED_LEVELS = frozenset({"easy", "medium", "hard", "level_4", "level_5"})


class OracleUnavailable(Exception):
    """Required observable history or validity evidence is unavailable."""


class OracleValidationFailure(Exception):
    """The reconstructed expert proposed an illegal action."""


def reconstruct_expert_state(prior_turns: list[dict]) -> dict:
    """Reconstruct memory from actual, valid executed learner actions."""

    if not isinstance(prior_turns, list):
        raise OracleUnavailable("missing prior turn history")

    state = new_security_expert_state()

    for row in prior_turns:
        if not isinstance(row, dict):
            raise OracleUnavailable("malformed history record")

        if (
            "executed_action" not in row
            or "executed_semantic_valid" not in row
        ):
            raise OracleUnavailable(
                "missing executed action or semantic-validity evidence"
            )

        info = row.get("info")
        if not isinstance(info, dict) or "valid" not in info:
            raise OracleUnavailable("missing recorded environment validity")

        semantic_valid = row["executed_semantic_valid"]
        environment_valid = info["valid"]

        if (
            type(semantic_valid) is not bool
            or type(environment_valid) is not bool
        ):
            raise OracleUnavailable("non-boolean validity evidence")

        action = row["executed_action"]
        if not isinstance(action, dict):
            raise OracleUnavailable("malformed executed action")

        # Invalid actions do not advance expert memory.
        if not (semantic_valid and environment_valid):
            continue

        kind = action.get("action_type")
        target = action.get("target", "")
        sub = action.get("sub_action", "none")

        if (
            not isinstance(kind, str)
            or kind not in {item.value for item in ActionType}
            or not isinstance(target, str)
            or not isinstance(sub, str)
        ):
            raise OracleUnavailable("malformed valid executed action")

        kind, target, sub = canonicalize_action(action)

        if kind == "investigate" and sub == "audit":
            state["audit_idx"] += 1

        elif kind == "monitor":
            state["monitor_idx"] += 1

        elif kind == "canary":
            if not target:
                raise OracleUnavailable("valid canary action missing target")
            state["canaried_departments"].add(target)

        elif kind == "neutralize" and sub == "turn":
            state["turned_sleeper"] = True

    return state


def _require_fields(record: dict, fields: set[str], label: str) -> None:
    """Reject incomplete records before Pydantic can fill defaults."""

    if not isinstance(record, dict):
        raise OracleUnavailable(f"malformed {label}")

    missing = fields - record.keys()
    if missing:
        raise OracleUnavailable(
            f"{label} missing required fields: {sorted(missing)}"
        )


def _label_observation_from_history(
    observation_before: dict,
    task_level: str,
    prior_turns: list[dict],
) -> tuple[str, str, str]:
    """Return the expert label using observable learner history only.

    The episode must start at turn 0 and contain every preceding turn.
    A hypothetical expert recommendation never advances expert memory.
    """

    _require_fields(
        observation_before,
        {
            "workers",
            "active_leaks",
            "canary_traps",
            "double_agents",
            "turn",
            "max_turns",
            "phase_number",
            "security_score",
        },
        "observation",
    )

    if type(task_level) is not str or task_level not in SUPPORTED_LEVELS:
        raise OracleUnavailable("unsupported task level")

    current_turn = observation_before["turn"]

    if type(current_turn) is not int or current_turn < 0:
        raise OracleUnavailable("invalid current turn")

    if not isinstance(prior_turns, list):
        raise OracleUnavailable("missing prior turn history")

    if len(prior_turns) != current_turn:
        raise OracleUnavailable("incomplete learner history")

    # Check actual turn indices, not just the number of records.
    for expected_turn, row in enumerate(prior_turns):
        if not isinstance(row, dict):
            raise OracleUnavailable("malformed history record")

        if type(row.get("turn")) is not int:
            raise OracleUnavailable("missing or invalid history turn")

        if row["turn"] != expected_turn:
            raise OracleUnavailable(
                f"non-contiguous history at turn {expected_turn}"
            )

        if not isinstance(row.get("executed_action"), dict):
            raise OracleUnavailable("malformed executed action")

    # Check observable fields used by the deterministic expert.
    nested_fields = {
        "workers": {
            "id",
            "name",
            "department",
            "state",
            "hire_turn",
            "suspicion_level",
            "turning_in_progress",
        },
        "active_leaks": {
            "id",
            "department",
            "is_canary",
            "verified",
            "turn_detected",
        },
        "canary_traps": {
            "id",
            "department",
        },
        "double_agents": {
            "worker_id",
            "active",
            "hydra_trust",
            "effectiveness",
            "disinfo_fed_count",
        },
    }

    for collection, required in nested_fields.items():
        records = observation_before[collection]

        if not isinstance(records, list):
            raise OracleUnavailable(f"malformed {collection}")

        for index, record in enumerate(records):
            _require_fields(
                record,
                required,
                f"{collection}[{index}]",
            )

    if (
        not observation_before["workers"]
        and not observation_before["canary_traps"]
    ):
        raise OracleUnavailable("no observable departments")

    try:
        observation = EnvironmentObservation.model_validate(
            observation_before
        )
    except (ValueError, TypeError) as exc:
        raise OracleUnavailable("invalid observation schema") from exc

    # Memory depends exclusively on recorded learner actions.
    state = reconstruct_expert_state(prior_turns)

    # The expert can mutate this copy without changing replay state.
    proposed_action = choose_security_first_action(
        observation,
        task_level,
        deepcopy(state),
    )

    legal, reason = validate_action(proposed_action, observation)

    if not legal:
        raise OracleValidationFailure(
            f"Expert proposed an illegal action: {reason}"
        )

    return canonicalize_action(proposed_action.model_dump())


_REPLAY_IDENTITY_FIELDS = (
    "synthetic",
    "experiment_id",
    "run_fingerprint",
    "checkpoint_sha256",
    "source_commit",
    "feature_extractor_version",
    "episode_id",
    "seed",
    "level",
)


def label_learner_turn(
    current_row: dict,
    task_level: str,
    prior_turns: list[dict],
    *,
    episode_header: dict | None = None,
    expected_identity: dict | None = None,
) -> tuple[str, str, str]:
    """Identity-checked replay entry point.

    expected_identity must eventually come from independently verified
    provenance. Equality checks alone do not authenticate artifact bytes.
    Real-data replay stays disabled until the integrated gate exists.
    """
    if type(expected_identity) is not dict:
        raise OracleUnavailable("missing independent replay identity")

    missing = set(_REPLAY_IDENTITY_FIELDS) - expected_identity.keys()
    if missing:
        raise OracleUnavailable(
            f"incomplete replay identity: {sorted(missing)}"
        )

    identity = {
        key: expected_identity[key]
        for key in _REPLAY_IDENTITY_FIELDS
    }

    for key in (
        "experiment_id",
        "feature_extractor_version",
        "episode_id",
    ):
        value = identity[key]
        if (
            type(value) is not str
            or not value.strip()
            or value != value.strip()
        ):
            raise OracleUnavailable(f"invalid replay identity: {key}")

    for key in ("run_fingerprint", "checkpoint_sha256"):
        value = identity[key]
        if (
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise OracleUnavailable(f"invalid replay identity: {key}")

    source = identity["source_commit"]
    if (
        type(source) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source) is None
    ):
        raise OracleUnavailable("invalid replay identity: source_commit")

    if type(identity["seed"]) is not int or identity["seed"] < 0:
        raise OracleUnavailable("invalid replay identity: seed")

    if type(identity["level"]) is not str or identity["level"] not in SUPPORTED_LEVELS:
        raise OracleUnavailable("unsupported replay level")

    if type(identity["synthetic"]) is not bool:
        raise OracleUnavailable("invalid synthetic classification")

    if identity["synthetic"] is not True:
        raise OracleUnavailable(
            "real replay requires the integrated provenance gate"
        )

    if task_level != identity["level"]:
        raise OracleUnavailable("task-level identity mismatch")

    header_result = validate_episode_header(
        episode_header, identity
    )
    if header_result["status"] != "accepted":
        raise OracleUnavailable(
            f"episode-header identity mismatch: {header_result['reasons']}"
        )

    if type(current_row) is not dict or type(prior_turns) is not list:
        raise OracleUnavailable("missing or malformed replay records")

    observation = current_row.get("observation_before")
    if type(observation) is not dict:
        raise OracleUnavailable("missing current observation_before")

    current_turn = observation.get("turn")
    if type(current_turn) is not int or current_turn < 0:
        raise OracleUnavailable("invalid current observation turn")

    if current_turn != len(prior_turns):
        raise OracleUnavailable("incomplete learner history")

    all_rows = [*prior_turns, current_row]

    for index, row in enumerate(all_rows):
        if type(row) is not dict:
            raise OracleUnavailable(
                f"malformed replay row at index {index}"
            )

        result = validate_episode_header(row, identity)
        if result["status"] != "accepted":
            raise OracleUnavailable(
                f"row identity mismatch at index {index}: "
                f"{result['reasons']}"
            )

        before = row.get("observation_before")
        if (
            type(before) is not dict
            or type(before.get("turn")) is not int
            or before["turn"] != index
        ):
            raise OracleUnavailable(
                f"unverifiable observation at index {index}"
            )

    continuity = validate_episode_turns(
        all_rows,
        expected_turn_count=current_turn + 1,
    )
    if continuity["status"] != "accepted":
        raise OracleUnavailable(
            f"invalid replay sequence: {continuity['reason']}"
        )

    return _label_observation_from_history(
        observation, task_level, prior_turns
    )
