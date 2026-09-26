"""Provisional synthetic-only Stage 1 feature candidates.

Not approved for real-artifact analysis. These low-level
helpers do not authenticate provenance; an integrated gate
is required before any future real-data entry point.
"""

import re

CANDIDATE_FEATURE_VERSION = "stage1-synthetic-feature-candidates-v1"
TRUNCATION_MARKER = "[... compacted for training context ...]"


def candidate_header(text):
    if not isinstance(text, str):
        return {"status": "unavailable", "reason": "invalid_text"}

    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    first_line = text.splitlines()[0] if text else ""
    match = re.fullmatch(
        r"Turn (\d+)/(\d+) \| Phase: (.+?) \((\d+)\)"
        r" \| Revenue: (-?\d+) \| Security: (-?\d+)",
        first_line,
    )

    if match is None:
        return {"status": "unavailable", "reason": "missing_header"}

    turn, maximum, phase, number, revenue, security = (
        match.groups()
    )
    return {
        "status": "available",
        "features": {
            "turn": int(turn),
            "max_turns": int(maximum),
            "phase": phase,
            "phase_number": int(number),
            "displayed_revenue": int(revenue),
            "displayed_security": int(security),
        },
    }


def candidate_triggered_canaries(text):
    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    match = re.search(
        r"^Canary Traps \((\d+)\):$",
        text,
        re.MULTILINE,
    )
    if match is None:
        return {
            "status": "unavailable",
            "reason": "missing_canary_section",
        }

    declared_count = int(match.group(1))
    rows = re.findall(
        r"^  canary-\d+ dept=\S+ triggered=(True|False)$",
        text,
        re.MULTILINE,
    )

    if len(rows) != declared_count:
        return {
            "status": "unavailable",
            "reason": "incomplete_canary_rows",
        }

    return {
        "status": "available",
        "triggered_count": rows.count("True"),
    }


def candidate_workforce_summary(text):
    """Conservative synthetic candidate; not the approved extractor."""
    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    lines = text.splitlines()
    workers = [
        (i, re.fullmatch(r"Workers \((\d+)\):", line))
        for i, line in enumerate(lines)
        if re.fullmatch(r"Workers \((\d+)\):", line)
    ]
    leaks = [
        i for i, line in enumerate(lines)
        if re.fullmatch(r"Active Leaks \(\d+\):", line)
    ]

    if len(workers) != 1 or len(leaks) != 1:
        return {"status": "unavailable", "reason": "missing_sections"}

    start, heading = workers[0]
    end = leaks[0]

    if end <= start:
        return {"status": "unavailable", "reason": "section_order"}

    declared = int(heading.group(1))
    worker_lines = lines[start + 1:end]

    omitted_matches = re.findall(
        r"^  clean loyal workers omitted: (\d+)$",
        text,
        re.MULTILINE,
    )
    if len(omitted_matches) > 1:
        return {"status": "unavailable", "reason": "invalid_omissions"}

    omitted = int(omitted_matches[0]) if omitted_matches else 0

    if worker_lines == ["  (none)"] and declared == 0:
        worker_lines = []

    pattern = re.compile(
        r"^  w-[0-9]+ .+? dept=(\S+) state=\S+ "
        r"(?:clean|suspicion=(\d+)%)(?: turning=\d+/4)?$"
    )
    parsed = [pattern.fullmatch(line) for line in worker_lines]

    if any(match is None for match in parsed):
        return {"status": "unavailable", "reason": "invalid_worker_rows"}

    if len(parsed) + omitted != declared:
        return {"status": "unavailable", "reason": "incomplete_worker_rows"}

    features = {
        "worker_count": declared,
        "displayed_high_suspicion_count": sum(
            match.group(2) is not None
            and int(match.group(2)) > 50
            for match in parsed
        ),
    }

    result = {"status": "available", "features": features}

    if omitted:
        result["unavailable_fields"] = ["unique_worker_departments"]
    else:
        features["unique_worker_departments"] = len({
            match.group(1) for match in parsed
        })

    return result


def candidate_learner_features(observation_before):
    """Synthetic candidate using only explicitly observable fields."""
    import math

    unavailable = {
        "status": "unavailable",
        "reason": "missing_or_invalid_learner_fields",
    }

    if type(observation_before) is not dict:
        return unavailable

    required = {
        "turn", "max_turns", "phase", "phase_number",
        "enterprise_revenue", "security_score",
        "workers", "canary_traps",
    }

    if not required.issubset(observation_before):
        return unavailable

    record = observation_before

    for name in ("turn", "max_turns", "phase_number"):
        if type(record[name]) is not int or record[name] < 0:
            return unavailable

    if type(record["phase"]) is not str or not record["phase"]:
        return unavailable

    for name in ("enterprise_revenue", "security_score"):
        value = record[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            return unavailable

    if type(record["workers"]) is not list:
        return unavailable

    if type(record["canary_traps"]) is not list:
        return unavailable

    departments = set()
    high_suspicion = 0

    for worker in record["workers"]:
        if type(worker) is not dict:
            return unavailable

        fields = {
            "id", "name", "department", "state",
            "suspicion_level", "turning_in_progress",
        }
        if not fields.issubset(worker):
            return unavailable

        for name in ("id", "name", "department", "state"):
            if type(worker[name]) is not str or not worker[name]:
                return unavailable

        if type(worker["turning_in_progress"]) is not bool:
            return unavailable

        suspicion = worker["suspicion_level"]
        if (
            type(suspicion) not in (int, float)
            or not math.isfinite(suspicion)
            or not 0 <= suspicion <= 1
        ):
            return unavailable

        departments.add(worker["department"])

        # Match the existing formatter's displayed percentage.
        if suspicion > 0.05:
            displayed = int(f"{suspicion:.0%}"[:-1])
            high_suspicion += displayed > 50

    triggered = 0

    for trap in record["canary_traps"]:
        if type(trap) is not dict:
            return unavailable

        if type(trap.get("triggered")) is not bool:
            return unavailable

        triggered += trap["triggered"] is True

    return {
        "status": "available",
        "header": {
            "turn": record["turn"],
            "max_turns": record["max_turns"],
            "phase": record["phase"],
            "phase_number": record["phase_number"],
            "displayed_revenue": int(
                f"{record['enterprise_revenue']:.0f}"
            ),
            "displayed_security": int(
                f"{record['security_score']:.0f}"
            ),
        },
        "workforce": {
            "worker_count": len(record["workers"]),
            "displayed_high_suspicion_count": high_suspicion,
            "unique_worker_departments": len(departments),
        },
        "triggered_canaries": triggered,
    }


def candidate_text_leak_asset_features(text):
    """Synthetic candidate; the feature allowlist is not yet approved."""
    if type(text) is not str:
        return {"status": "unavailable", "reason": "invalid_text"}

    if TRUNCATION_MARKER in text:
        return {"status": "unavailable", "reason": "token_truncated"}

    lines = text.splitlines()

    def section(title):
        matches = [
            (index, re.fullmatch(
                rf"{re.escape(title)} \((\d+)\):", line
            ))
            for index, line in enumerate(lines)
            if re.fullmatch(
                rf"{re.escape(title)} \((\d+)\):", line
            )
        ]

        if len(matches) != 1:
            return None

        start, heading = matches[0]
        declared = int(heading.group(1))
        rows = []

        for line in lines[start + 1:]:
            if not line.startswith("  "):
                break

            # Compaction appends this worker summary after the
            # original final section, which is double agents.
            if (
                title == "Active Double Agents"
                and re.fullmatch(
                    r"  clean loyal workers omitted: \d+", line
                )
            ):
                continue

            rows.append(line)

        if declared == 0:
            return [] if rows == ["  (none)"] else None

        return rows if len(rows) == declared else None

    leak_rows = section("Active Leaks")
    asset_rows = section("Active Double Agents")

    if leak_rows is None or asset_rows is None:
        return {
            "status": "unavailable",
            "reason": "incomplete_leak_or_asset_section",
        }

    leak_pattern = re.compile(
        r"  \S+ dept=\S+ channel=\S+"
        r"( \[CANARY MATCH\])?"
    )
    asset_pattern = re.compile(
        r"  \S+ active=(True|False)"
        r" trust=\d+% eff=\d+% disinfo=\d+"
    )

    leaks = [leak_pattern.fullmatch(row) for row in leak_rows]
    assets = [asset_pattern.fullmatch(row) for row in asset_rows]

    if any(match is None for match in leaks + assets):
        return {
            "status": "unavailable",
            "reason": "malformed_leak_or_asset_row",
        }

    return {
        "status": "available",
        "features": {
            "active_leak_count": len(leaks),
            "displayed_canary_match_count": sum(
                match.group(1) is not None for match in leaks
            ),
            "displayed_double_agent_count": len(assets),
            "operational_double_agent_count": sum(
                match.group(1) == "True" for match in assets
            ),
        },
    }


def candidate_learner_leak_asset_features(observation_before):
    unavailable = {
        "status": "unavailable",
        "reason": "invalid_learner_leak_or_asset_fields",
    }

    if type(observation_before) is not dict:
        return unavailable

    leaks = observation_before.get("active_leaks")
    assets = observation_before.get("double_agents")

    if type(leaks) is not list or type(assets) is not list:
        return unavailable

    if any(
        type(row) is not dict
        or type(row.get("is_canary")) is not bool
        for row in leaks
    ):
        return unavailable

    if any(
        type(row) is not dict
        or type(row.get("active")) is not bool
        for row in assets
    ):
        return unavailable

    return {
        "status": "available",
        "features": {
            "active_leak_count": len(leaks),
            "displayed_canary_match_count": sum(
                row["is_canary"] for row in leaks
            ),
            "displayed_double_agent_count": len(assets),
            "operational_double_agent_count": sum(
                row["active"] for row in assets
            ),
        },
    }
