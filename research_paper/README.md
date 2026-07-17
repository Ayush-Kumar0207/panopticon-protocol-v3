# Panopticon Protocol Research Paper Package

This is a self-contained, venue-neutral research workspace for turning Panopticon Protocol v3 into a professional paper and reproducibility artifact.

The package is deliberately honest about research maturity. It contains a complete manuscript draft, verified citations, generated tables and figures, evaluation and statistical-analysis protocols, dataset/model documentation, ethics and limitations statements, reviewer/submission templates, and validation scripts. The checked-in V5 results are **preliminary evidence**, not yet the final statistical study: they use 20 episodes per agent–level and the raw V5 model fails the advanced security release gate.

## Proposed paper

**Working title:** *Panopticon Protocol: Security-Gated Evaluation of Learned Agents in a Partially Observable Counter-Espionage Environment*

**Central research claim:** A composite reward or average grade can hide operationally unsafe behavior in long-horizon agent environments; explicit security gates reveal this failure, and a deterministic supervisor can establish task solvability without being misreported as raw learned-policy performance.

The paper does **not** claim that the current raw fine-tuned model is safety-complete. Its strongest defensible result is that raw V5 improves the matched macro grade from `0.641110` to `0.701627` but fails nine advanced-tier acceptance checks, whereas the separately labeled security-first supervisor reaches `0.790471` and passes the checked-in gate.

## Directory map

```text
research_paper/
├── paper/                 LaTeX manuscript, supplement, author metadata, BibTeX
├── assets/figures/        Generated paper figures plus preserved source plots
├── assets/tables/         Machine-readable and LaTeX result tables
├── data/processed/        Compact metrics derived from checked-in evaluation JSON
├── data/seed_plans/       Digest-protected pilot/development/final V6 seeds
├── data/raw/              Raw-data index; large artifacts remain at repository root
├── protocols/             Evaluation, statistics, ablation, ethics, cards, reproducibility
├── scripts/               Deterministic extraction, plotting, validation, and build helpers
├── templates/             Cover letter and reviewer-response templates
├── checklists/            Submission and reporting checklists
├── CITATION.cff           Citation metadata for the artifact
├── MANIFEST.md            Purpose and provenance of every asset class
└── RESEARCH_STATUS.md      What is complete and what must happen before submission
```

## Rebuild the evidence package

From the repository root on Windows PowerShell:

```powershell
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\extract_metrics.py
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\generate_figures.py
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\generate_diagrams.py
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\reconstruct_training_seed_ledger.py --overwrite
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\verify_drive_seed_ledger.py
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\verify_seed_separation.py
& '.\.venv-infer\Scripts\python.exe' research_paper\scripts\validate_package.py
```

The scripts use only repository-local inputs. `extract_metrics.py` reads `evaluation_comparison_latest.json`, writes compact processed data and tables, and records hashes of the large evidence files. `generate_figures.py` regenerates the paper-specific plots. The seed scripts bind the reconstruction to the exact training commit, compare every ordered seed with the five original Drive-side expert-metrics files, and prove zero overlap with all frozen V6 splits. The directly verified 250-seed ledger has digest `741276b1fcbab159db7fee95d5e418f91a12316f6ca4861479f78af595c4415d`. `validate_package.py` checks required files, result consistency, citation keys, seed evidence, unresolved placeholders, and figure presence.

## Run the V6 provenance-aware harness

The V6 runner stores raw and executed actions separately, categorizes repairs, hashes compact traces, checkpoints every episode to JSONL, and binds a run to the frozen seed-plan digest:

```powershell
python v6_evaluation.py `
  --split pilot `
  --policies random,heuristic,security_first,model_raw,model_repair `
  --output-dir research_paper\data\v6_runs\pilot
```

The original HYDRA is `scripted_memory_v1`: adaptive rules, not a neural network. A separate trainable 12,685-parameter neural HYDRA is implemented in `hydra_neural.py`/`train_hydra.py`. Its held-out comparison is governed by `protocols/learned_hydra_protocol.md`; no effectiveness claim is currently made.

## Run the complete Colab experiment

Use [`../Panopticon_V6_Research_Colab.ipynb`](../Panopticon_V6_Research_Colab.ipynb) for the remaining GPU work and [`COLAB_V6_BEGINNER_RUNBOOK.md`](COLAB_V6_BEGINNER_RUNBOOK.md) for cell-by-cell instructions. Neural training is restricted to the frozen development split, saves one rolling atomic checkpoint after every episode, restores RNG and optimizer state exactly, and never accumulates historical checkpoint directories. Scripted/neural pilot and final evaluation append one durable record per episode and resume by verified episode key.

The final cell is deliberately locked until the pilot is complete and the protocol is frozen.

## Build the manuscript

A TeX distribution is not currently installed in this workspace. After installing TeX Live or MiKTeX:

```powershell
Set-Location research_paper\paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, upload the `research_paper` folder to Overleaf and set `paper/main.tex` as the main document. The manuscript intentionally uses common LaTeX packages.

## Before submitting anywhere

Read [RESEARCH_STATUS.md](RESEARCH_STATUS.md). At minimum, the authors must:

1. choose a venue and apply its official style/page limits;
2. confirm names, affiliations, email addresses, ORCIDs, acknowledgements, funding, and conflicts;
3. run the preregistered extended held-out evaluation and planned ablations;
4. regenerate tables/figures from final artifacts and freeze hashes;
5. obtain an independent reproduction or second-person audit; and
6. replace every `AUTHOR-TODO` marker.

## Reporting rule

Always report these as different systems:

- **raw V5 model** — merged LoRA model without deterministic security supervision;
- **repaired model output** — model output after parsing or semantic repair;
- **security-first supervisor** — deterministic control policy;
- **hybrid system** — model plus any supervisor/fallback.

Supervisor performance must never be attributed to the raw neural policy.
