# Artifact Manifest

## Authored research assets

| Path | Purpose |
|---|---|
| `paper/main.tex` | Main venue-neutral manuscript. |
| `paper/supplementary.tex` | Environment details, formulas, and known gaps. |
| `paper/author_metadata.tex` | Author metadata and anonymous-review switch. |
| `paper/references.bib` | Foundational RL/agent bibliography. |
| `paper/references_security.bib` | Agent-security and red-teaming bibliography. |
| `paper/ABSTRACT_AND_TITLE_OPTIONS.md` | Title, short abstract, highlights, and lay summary. |
| `protocols/research_questions.md` | Research questions, hypotheses, outcomes, and falsification. |
| `protocols/evaluation_protocol.md` | Frozen benchmark design and logging requirements. |
| `protocols/statistical_analysis_plan.md` | Pre-result statistical plan. |
| `protocols/ablation_plan.md` | Component-level experiments and interpretation rules. |
| `protocols/reproducibility.md` | End-to-end reproduction instructions. |
| `protocols/learned_hydra_protocol.md` | Preregistered scripted-versus-neural adversary study. |
| `protocols/claim_evidence_matrix.csv` | Claims mapped to code/data evidence and allowed wording. |
| `protocols/dataset_card.md` | Expert-trajectory dataset documentation. |
| `protocols/model_card.md` | Base, raw, supervisor, and hybrid policy documentation. |
| `protocols/ethics_and_dual_use.md` | Ethics, dual-use, and responsible-release controls. |
| `protocols/limitations_and_threats.md` | Construct/internal/statistical/external validity threats. |
| `protocols/standards_mapping.md` | OWASP, MITRE ATLAS, and NIST coverage crosswalk. |
| `protocols/research_to_product.md` | Evidence-gated commercialization strategy. |
| `protocols/data_management_plan.md` | Retention, integrity, privacy, and archival plan. |
| `templates/cover_letter.md` | Submission cover-letter template. |
| `templates/reviewer_response.md` | Point-by-point response template. |
| `templates/pilot_evaluation_report.md` | Authorized commercial-pilot report template. |
| `checklists/submission_checklist.md` | Scientific, artifact, ethics, and administrative checks. |
| `checklists/author_contributions.md` | CRediT contribution worksheet. |
| `COLAB_V6_BEGINNER_RUNBOOK.md` | Beginner-safe GPU execution, interruption recovery, and final-split instructions. |

## Generated assets

| Path | Generator | Source |
|---|---|---|
| `data/processed/headline_metrics.json` | `scripts/extract_metrics.py` | `evaluation_comparison_latest.json` |
| `data/processed/per_level_metrics.csv` | `scripts/extract_metrics.py` | same |
| `assets/tables/macro_results.csv` | `scripts/extract_metrics.py` | same |
| `assets/tables/macro_results.tex` | `scripts/extract_metrics.py` | same |
| `assets/tables/advanced_gate_results.csv` | `scripts/extract_metrics.py` | same |
| `assets/figures/macro_grade.{png,pdf}` | `scripts/generate_figures.py` | compact comparison |
| `assets/figures/per_level_grade.{png,pdf}` | `scripts/generate_figures.py` | compact comparison |
| `assets/figures/security_revenue_tradeoff.{png,pdf}` | `scripts/generate_figures.py` | compact comparison |
| `assets/figures/advanced_pass_rate.{png,pdf}` | `scripts/generate_figures.py` | compact comparison |
| `assets/figures/training_data_by_level.{png,pdf}` | `scripts/generate_figures.py` | training metadata |
| `assets/figures/research_architecture.{png,pdf}` | `scripts/generate_diagrams.py` | implemented architecture |
| `assets/figures/research_to_product_roadmap.{png,pdf}` | `scripts/generate_diagrams.py` | commercialization plan |
| `data/raw/SHA256SUMS.txt` | `scripts/extract_metrics.py` | repository evidence files |
| `data/seed_plans/v6_seed_plan.json` | `scripts/create_seed_plan.py` | master seed plus optional training ledger |
| `data/training_seed_ledger.reconstructed.json` | `scripts/reconstruct_training_seed_ledger.py` | V5 compact metadata plus deterministic generator code |
| `data/raw/v5_drive_seed_evidence.json` | Authenticated read-only Drive extraction | ordered seeds and identities from five original expert-metrics files |
| `data/training_seed_ledger.drive_verified.json` | `scripts/verify_drive_seed_ledger.py` | exact Drive-versus-reconstruction comparison |
| `data/seed_plans/v6_training_separation_report.json` | `scripts/verify_seed_separation.py` | frozen V6 plan plus directly verified V5 ledger |
| `../Panopticon_V6_Research_Colab.ipynb` | `scripts/build_colab_v6_notebook.py` | complete pinned, resume-safe pilot/training/final Colab workflow |

## Implemented research extension

| Path | Purpose |
|---|---|
| `../panopticon_bench/` | Versioned long-horizon campaign, adapter, runner, provenance, metric, and seed-plan package. |
| `../v6_evaluation.py` | Frozen-seed, append-only, provenance-aware evaluation harness. |
| `../hydra_policy.py` | Typed HYDRA boundary and original scripted-memory baseline. |
| `../hydra_neural.py` | Versioned trainable neural HYDRA policy and checkpoint format. |
| `../train_hydra.py` | Resumable simulator-only policy-gradient training entry point. |
| `../panopticon_bench/scenarios/` | Non-destructive synthetic-canary and approval-boundary fixtures. |
| `../tests/test_panopticon_bench.py` | Schema, authorization, runner, CMG, and attribution tests. |
| `../tests/test_research_validity.py` | Action-mask, target-coverage, redaction, and HYDRA policy tests. |
| `../tests/test_server_hardening.py` | Privileged-route, public-observation, and CORS-default tests. |
| `../tests/test_packaging_contract.py` | Docker source and application-entry-point contract tests. |
| `../.github/workflows/research-artifact.yml` | Clean CI regeneration and validation. |

## Preserved source plots

Files prefixed `assets/figures/source_` are copied from the existing `plots/` directory for historical provenance. They are not automatically treated as final paper figures.

## Large raw artifacts

The 238 MB `evaluationResults.json` and 9 MB `training_events_fixed_ep20.jsonl` remain at the repository root to avoid duplication. `data/raw/README.md` describes them, while `SHA256SUMS.txt` identifies exact local bytes. A public artifact may archive these separately.
