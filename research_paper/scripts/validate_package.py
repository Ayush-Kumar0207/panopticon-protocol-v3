"""Validate research-package integrity and headline claim consistency."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research_paper"

REQUIRED = [
    "README.md",
    "RESEARCH_STATUS.md",
    "MANIFEST.md",
    "COLAB_V6_BEGINNER_RUNBOOK.md",
    "scripts/build_colab_v6_notebook.py",
    "CITATION.cff",
    "paper/main.tex",
    "paper/supplementary.tex",
    "paper/author_metadata.tex",
    "paper/references.bib",
    "paper/references_security.bib",
    "paper/ABSTRACT_AND_TITLE_OPTIONS.md",
    "protocols/research_questions.md",
    "protocols/evaluation_protocol.md",
    "protocols/statistical_analysis_plan.md",
    "protocols/ablation_plan.md",
    "protocols/reproducibility.md",
    "protocols/learned_hydra_protocol.md",
    "protocols/claim_evidence_matrix.csv",
    "protocols/dataset_card.md",
    "protocols/model_card.md",
    "protocols/ethics_and_dual_use.md",
    "protocols/limitations_and_threats.md",
    "protocols/standards_mapping.md",
    "protocols/research_to_product.md",
    "protocols/data_management_plan.md",
    "data/processed/headline_metrics.json",
    "data/processed/per_level_metrics.csv",
    "data/raw/SHA256SUMS.txt",
    "data/seed_plans/v6_seed_plan.json",
    "data/training_seed_ledger.reconstructed.json",
    "data/training_seed_ledger.drive_verified.json",
    "data/raw/v5_drive_seed_evidence.json",
    "data/seed_plans/v6_training_separation_report.json",
    "assets/tables/macro_results.csv",
    "assets/tables/macro_results.tex",
    "assets/tables/advanced_gate_results.csv",
]

FIGURES = [
    "macro_grade",
    "per_level_grade",
    "security_revenue_tradeoff",
    "advanced_pass_rate",
    "training_data_by_level",
    "research_architecture",
    "research_to_product_roadmap",
]

EXPECTED_MACRO = {
    "base_untrained": 0.641110,
    "raw_v5_trained": 0.701627,
    "security_first_supervisor": 0.790471,
    "heuristic": 0.689400,
    "random": 0.647400,
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED:
        path = PACKAGE / rel
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing or empty required file: {rel}", errors)

    notebook_path = ROOT / "Panopticon_V6_Research_Colab.ipynb"
    if not notebook_path.is_file():
        fail("missing generated Panopticon V6 Colab notebook", errors)
    else:
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            source = "".join(
                "".join(cell.get("source", [])) for cell in cells
            )
            if notebook.get("nbformat") != 4 or len(cells) < 20:
                fail("V6 Colab notebook has an invalid format or incomplete cell set", errors)
            for cell_number, cell in enumerate(cells, start=1):
                if cell.get("cell_type") == "code":
                    ast.parse("".join(cell.get("source", [])), filename=f"cell-{cell_number}")
            required_notebook_markers = [
                "research-v6-pilot-2026-07-17-r1",
                "HYDRA_CHECKPOINT_EVERY = 1",
                "I_UNDERSTAND_FINAL_SPLIT_IS_SINGLE_USE",
                "--resume",
                "transformers==4.57.6",
                "console_logs",
                "Last child-process output",
            ]
            for marker in required_notebook_markers:
                if marker not in source:
                    fail(f"V6 Colab notebook is missing marker: {marker}", errors)
        except Exception as exc:
            fail(f"invalid V6 Colab notebook: {exc}", errors)

    seed_plan_path = PACKAGE / "data" / "seed_plans" / "v6_seed_plan.json"
    if seed_plan_path.exists():
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from panopticon_bench.seed_plan import canonical_sha256, load_seed_plan, validate_seed_plan

            seed_plan = load_seed_plan(seed_plan_path)
            ledger_path = PACKAGE / "data" / "training_seed_ledger.drive_verified.json"
            reconstructed_path = PACKAGE / "data" / "training_seed_ledger.reconstructed.json"
            report_path = PACKAGE / "data" / "seed_plans" / "v6_training_separation_report.json"
            if ledger_path.exists() and reconstructed_path.exists() and report_path.exists():
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                reconstructed = json.loads(reconstructed_path.read_text(encoding="utf-8"))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                training_seeds = [int(seed) for seed in ledger["seeds"]]
                validate_seed_plan(seed_plan, excluded_training_seeds=training_seeds)
                if ledger.get("seeds_sha256") != canonical_sha256(training_seeds):
                    fail("Drive-verified training-seed ledger digest mismatch", errors)
                if ledger.get("seeds") != reconstructed.get("seeds"):
                    fail("Drive-verified and reconstructed training ledgers differ", errors)
                if not str(ledger.get("verification_status", "")).startswith("directly-verified"):
                    fail("training ledger is not marked directly verified", errors)
                if report.get("seed_plan_sha256") != seed_plan["seed_plan_sha256"]:
                    fail("training-separation report references the wrong seed plan", errors)
                expected_ledger_path = ledger_path.relative_to(ROOT).as_posix()
                if report.get("training_ledger_path") != expected_ledger_path:
                    fail("training-separation report references the wrong ledger path", errors)
                if report.get("training_ledger_seeds_sha256") != ledger.get("seeds_sha256"):
                    fail("training-separation report references the wrong training ledger", errors)
                if report.get("overlap_count") != 0:
                    fail("training-separation report records a non-zero overlap", errors)
                if report.get("conclusion") != "directly-verified-disjoint":
                    fail("training-separation report is not directly verified", errors)
            elif seed_plan["training_separation_status"].startswith("unverified"):
                warnings.append("V6 seed plan has not been checked against a training-seed ledger")
        except Exception as exc:
            fail(f"invalid V6 seed plan: {exc}", errors)

    source_path = ROOT / "evaluation_comparison_latest.json"
    processed_path = PACKAGE / "data" / "processed" / "headline_metrics.json"
    if source_path.exists() and processed_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        processed = json.loads(processed_path.read_text(encoding="utf-8"))
        for agent, expected in EXPECTED_MACRO.items():
            actual = source["level_macro_averages"][agent]["grade"]
            generated = processed["agents"][agent]["macro"]["grade"]
            if abs(actual - expected) > 1e-9:
                fail(f"source macro drift for {agent}: {actual} != {expected}", errors)
            if abs(generated - actual) > 1e-12:
                fail(f"generated macro mismatch for {agent}", errors)
        failed = len(source["acceptance"]["raw_v5_trained"]["failed_checks"])
        if failed != 9:
            fail(f"expected 9 raw V5 failed checks, found {failed}", errors)
        if not source["acceptance"]["security_first_supervisor"]["accepted"]:
            fail("supervisor acceptance unexpectedly false", errors)

    for stem in FIGURES:
        for suffix in (".png", ".pdf"):
            path = PACKAGE / "assets" / "figures" / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size < 1_000:
                fail(f"missing/small generated figure: {path.relative_to(PACKAGE)}", errors)

    tex_path = PACKAGE / "paper" / "main.tex"
    bib_paths = [
        PACKAGE / "paper" / "references.bib",
        PACKAGE / "paper" / "references_security.bib",
    ]
    if tex_path.exists() and all(path.exists() for path in bib_paths):
        tex = tex_path.read_text(encoding="utf-8")
        bib = "\n".join(path.read_text(encoding="utf-8") for path in bib_paths)
        used = citation_keys(tex)
        defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
        missing = sorted(used - defined)
        if missing:
            fail(f"undefined citation keys: {missing}", errors)
        if "not attributed to the raw model" not in tex:
            fail("main manuscript is missing the supervisor attribution disclaimer", errors)
        if "20 episodes" not in tex:
            fail("main manuscript is missing the preliminary sample-size disclosure", errors)

    hashes = PACKAGE / "data" / "raw" / "SHA256SUMS.txt"
    if hashes.exists():
        text = hashes.read_text(encoding="utf-8")
        names = ["evaluation_comparison_latest.json"]
        if (ROOT / "evaluationResults.json").exists():
            names.append("evaluationResults.json")
        for name in names:
            if name not in text:
                fail(f"hash manifest missing {name}", errors)

    todo_count = sum(
        path.read_text(encoding="utf-8", errors="replace").count("AUTHOR-TODO")
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".tex", ".json", ".cff"}
    )
    if todo_count:
        warnings.append(f"{todo_count} AUTHOR-TODO markers remain by design; resolve before submission")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAILED: {len(errors)} validation error(s)")
        return 1
    print("PASS: research package structure and headline claims are internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
