# Reproducibility Guide

## Reproducibility levels

- **Artifact regeneration:** Recreate compact metrics, tables, and figures from checked-in summaries. Available now.
- **Environment reproduction:** Run deterministic expert smoke/security tests from a clean Python environment. Available now.
- **Model evaluation reproduction:** Requires access to the exact base/merged checkpoint and enough inference compute.
- **Training reproduction:** Requires the base model, GPU resources, frozen expert-data seeds, and V5 training configuration.
- **Independent scientific replication:** A separate person executes the frozen final protocol without using the authors' final-test outcomes for tuning.

## 1. Repository setup

```powershell
git clone https://github.com/Ayush-Kumar0207/panopticon-protocol-v3.git
Set-Location panopticon-protocol-v3
python -m venv .venv-research
& '.\.venv-research\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv-research\Scripts\python.exe' -m pip install -e '.[train]'
```

For LLM training/evaluation, install the pinned versions recorded by the final experiment manifest; the root optional dependencies alone do not capture the full TRL/PEFT stack.

## 2. Environment verification

```powershell
& '.\.venv-research\Scripts\python.exe' smoke_test.py
& '.\.venv-research\Scripts\python.exe' security_regression_test.py
```

Expected current result: all five smoke-test tiers pass; the security regression passes 20/20 episodes at each of five tiers. These verify the deterministic expert path, not raw V5 performance.

## 3. Research-validity tests

```powershell
& '.\.venv-research\Scripts\python.exe' -m pytest -q tests\test_research_validity.py tests\test_panopticon_bench.py
```

These tests verify full worker target coverage, canonical semantic masking, masked-PPO probability recomputation, ARGUS hidden-state redaction, deterministic scripted HYDRA, and trainable neural-HYDRA decisions.

## 4. Freeze evaluation seeds

```powershell
& '.\.venv-research\Scripts\python.exe' research_paper\scripts\create_seed_plan.py `
  --training-seed-ledger research_paper\data\training_seeds.json
```

The checked-in V6 seed plan is internally disjoint and digest-protected. The original Drive-side metadata and five `expert_metrics_*.json` files were read through an authenticated, read-only connector. The compact evidence snapshot contains all 250 ordered episode seeds plus Drive file identities. These commands reproduce the three-stage check: reconstruct from the exact source commit, compare every value with Drive evidence, then test V6 disjointness.

```powershell
python research_paper\scripts\reconstruct_training_seed_ledger.py --overwrite
python research_paper\scripts\verify_drive_seed_ledger.py
python research_paper\scripts\verify_seed_separation.py
```

The direct and reconstructed ledgers match exactly with seed digest `741276b1fcbab159db7fee95d5e418f91a12316f6ca4861479f78af595c4415d`. The final separation report records 250 training seeds, 1,150 evaluation seeds, and zero overlap.

## 5. Regenerate compact paper artifacts

```powershell
& '.\.venv-research\Scripts\python.exe' research_paper\scripts\extract_metrics.py
& '.\.venv-research\Scripts\python.exe' research_paper\scripts\generate_figures.py
& '.\.venv-research\Scripts\python.exe' research_paper\scripts\validate_package.py
```

Compare `research_paper/data/raw/SHA256SUMS.txt` and generated outputs with the archived release.

## 6. Model identity

Record:

- Hugging Face repository plus immutable revision for the base model;
- tokenizer file hashes and chat-template hash;
- each adapter file hash;
- merged checkpoint hash/manifest;
- dtype/quantization and device mapping;
- generation library versions; and
- prompt, parser, repair, and supervisor hashes.

Do not identify a mutable directory path alone as a model version.

## 7. Dataset identity

The compact summary reports 250 accepted expert episodes and 88,896 weighted examples. A final release must archive:

- unweighted trajectory IDs and episode metadata;
- weighting rule and resulting sample multiplicities;
- training/development seed lists or their auditable encrypted commitments;
- trajectory schema version;
- tokenizer/max-length configuration; and
- checks proving final-test seed disjointness.

## 8. Learned HYDRA reproduction

`scripted_memory_v1` is the reported baseline. Neural HYDRA training is optional, compute-heavy, and must use only non-final seeds:

```powershell
& '.\.venv-research\Scripts\python.exe' train_hydra.py `
  --episodes 2000 `
  --seed 20260715 `
  --output checkpoints\hydra_neural_seed20260715.pt
```

Evaluate scripted and neural HYDRA in separate V6 output directories with the same frozen seed digest. Follow `learned_hydra_protocol.md`; a training score is never a held-out result.

## 9. Final evaluation

Follow `evaluation_protocol.md` and `statistical_analysis_plan.md`. Write append-only per-episode JSONL. After completion:

1. hash raw logs;
2. run the frozen analysis;
3. save software/hardware metadata;
4. regenerate paper numbers and figures;
5. prevent further checkpoint selection on final data; and
6. archive code/data/model artifacts under a versioned release and DOI where possible.

## 10. Manuscript build

```powershell
Set-Location research_paper\paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current workspace does not contain a TeX distribution, so manuscript source is validated structurally but not locally rendered here.

## 11. Reproduction report template

Record date, person, clean-machine identifier, source commit, package-lock hash, hardware, commands, unexpected deviations, test outputs, metric differences, and a signed conclusion: exact, numerically compatible, behaviorally compatible, or failed reproduction.
