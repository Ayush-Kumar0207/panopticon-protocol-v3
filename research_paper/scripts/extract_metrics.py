"""Extract compact, publication-facing metrics from the checked-in V5 summary.

This script intentionally does not parse the older 238 MB evaluationResults.json.
The paper draft is tied to evaluation_comparison_latest.json and records hashes of
larger artifacts for provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research_paper"
SOURCE = ROOT / "evaluation_comparison_latest.json"
PROCESSED = PACKAGE / "data" / "processed"
TABLES = PACKAGE / "assets" / "tables"
RAW_INDEX = PACKAGE / "data" / "raw"

LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]
AGENT_ORDER = [
    "base_untrained",
    "raw_v5_trained",
    "security_first_supervisor",
    "heuristic",
    "random",
]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source() -> dict[str, Any]:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source summary: {SOURCE}")
    with SOURCE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != "panopticon-security-v5-evaluation-comparison-v1":
        raise ValueError("Unexpected evaluation comparison schema; do not silently mix experiments")
    if data["benchmark"]["levels"] != LEVELS:
        raise ValueError("Unexpected level order")
    missing = set(AGENT_ORDER) - set(data["agents"])
    if missing:
        raise ValueError(f"Missing expected agents: {sorted(missing)}")
    return data


def acceptance_label(data: dict[str, Any], agent: str) -> str:
    record = data.get("acceptance", {}).get(agent)
    if record is None:
        return "n/a"
    return "passed" if record.get("accepted") else "failed"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_headline(data: dict[str, Any]) -> dict[str, Any]:
    macro = data["level_macro_averages"]
    return {
        "schema_version": "panopticon-paper-headline-metrics-v1",
        "source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "source_created_at": data["created_at"],
        "benchmark": data["benchmark"],
        "agents": {
            agent: {
                "label": data["agents"][agent]["label"],
                "macro": macro[agent],
                "acceptance": acceptance_label(data, agent),
            }
            for agent in AGENT_ORDER
        },
        "comparisons": data["comparisons"],
        "acceptance": data["acceptance"],
        "paper_claims": {
            "raw_v5_absolute_grade_gain": macro["raw_v5_trained"]["grade"]
            - macro["base_untrained"]["grade"],
            "raw_v5_relative_grade_gain_percent": 100
            * (
                macro["raw_v5_trained"]["grade"]
                - macro["base_untrained"]["grade"]
            )
            / macro["base_untrained"]["grade"],
            "raw_v5_failed_advanced_checks": len(
                data["acceptance"]["raw_v5_trained"]["failed_checks"]
            ),
        },
    }


def build_per_level_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in AGENT_ORDER:
        record = data["agents"][agent]
        for index, level in enumerate(LEVELS):
            row = {
                "agent": agent,
                "label": record["label"],
                "level": level,
                "episodes": data["benchmark"]["episodes_per_agent_level"],
                "grade": record["grade"][index],
                "grade_std": record["grade_std"][index],
                "reward": record["reward"][index],
                "revenue": record["revenue"][index],
                "security": record["security"][index],
                "sleepers_caught": record["sleepers_caught"][index],
                "pass_rate": "",
            }
            if "pass_rate" in record:
                row["pass_rate"] = record["pass_rate"][index]
            rows.append(row)
    return rows


def build_macro_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in AGENT_ORDER:
        macro = data["level_macro_averages"][agent]
        rows.append(
            {
                "agent": agent,
                "policy": data["agents"][agent]["label"],
                "macro_grade": f"{macro['grade']:.6f}",
                "macro_reward": f"{macro['reward']:.3f}",
                "macro_revenue": f"{macro['revenue']:.3f}",
                "macro_security": f"{macro['security']:.3f}",
                "macro_caught": f"{macro['sleepers_caught']:.3f}",
                "acceptance": acceptance_label(data, agent),
            }
        )
    return rows


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_macro_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Policy & Grade & Reward & Revenue & Security & Accepted \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['policy']))} & {row['macro_grade']} & "
            f"{row['macro_reward']} & {row['macro_revenue']} & "
            f"{row['macro_security']} & {row['acceptance']} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_advanced_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in ("raw_v5_trained", "security_first_supervisor"):
        record = data["agents"][agent]
        for level in ("level_4", "level_5"):
            index = LEVELS.index(level)
            rows.append(
                {
                    "agent": agent,
                    "policy": record["label"],
                    "level": level,
                    "episodes": data["benchmark"]["episodes_per_agent_level"],
                    "pass_rate": record["pass_rate"][index],
                    "security": record["security"][index],
                    "sleepers_caught": record["sleepers_caught"][index],
                    "candidate_accepted": acceptance_label(data, agent),
                }
            )
    return rows


def write_hashes() -> None:
    candidates = [
        ROOT / "evaluationResults.json",
        ROOT / "training_events_fixed_ep20.jsonl",
        SOURCE,
        ROOT / "evaluation_snapshot_apr26.json",
        ROOT / "plots" / "training_statistics.json",
    ]
    lines = []
    for path in candidates:
        if path.exists():
            lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    RAW_INDEX.mkdir(parents=True, exist_ok=True)
    (RAW_INDEX / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    data = load_source()
    headline = build_headline(data)
    per_level = build_per_level_rows(data)
    macro_rows = build_macro_rows(data)
    advanced_rows = build_advanced_rows(data)

    write_json(PROCESSED / "headline_metrics.json", headline)
    write_csv(
        PROCESSED / "per_level_metrics.csv",
        list(per_level[0].keys()),
        per_level,
    )
    write_csv(TABLES / "macro_results.csv", list(macro_rows[0].keys()), macro_rows)
    write_macro_latex(TABLES / "macro_results.tex", macro_rows)
    write_csv(
        TABLES / "advanced_gate_results.csv",
        list(advanced_rows[0].keys()),
        advanced_rows,
    )
    write_hashes()
    print(f"Wrote processed metrics for {len(AGENT_ORDER)} agents and {len(LEVELS)} levels")
    print(
        "Raw V5 grade gain: "
        f"{headline['paper_claims']['raw_v5_absolute_grade_gain']:.6f} absolute; "
        f"{headline['paper_claims']['raw_v5_relative_grade_gain_percent']:.2f}% relative"
    )
    print(
        "Raw V5 failed checks: "
        f"{headline['paper_claims']['raw_v5_failed_advanced_checks']}"
    )


if __name__ == "__main__":
    main()
