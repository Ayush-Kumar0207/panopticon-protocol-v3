# Research Status and Submission Gates

Last updated: 2026-07-15

## Current status

The **paper package is complete as a professional research scaffold and first manuscript draft**. The **scientific study is not yet submission-ready** because the final high-powered evaluation and ablation matrix have not been run. This distinction prevents formatting completeness from being confused with evidential completeness.

## Green — complete now

- A defensible research question and contribution statement.
- A full venue-neutral LaTeX manuscript and supplement.
- Related-work bibliography based on primary papers.
- Exact current V5 headline results transcribed through a deterministic script.
- Figure and table generation from checked-in JSON.
- Claim-to-evidence matrix.
- Evaluation, statistical-analysis, ablation, and reproducibility protocols.
- Dataset card, model card, ethics statement, limitations, and threat model.
- Artifact manifest, raw-data index, and SHA-256 tracking.
- Cover-letter, reviewer-response, and submission-checklist templates.
- A validator that fails on inconsistent headline numbers and missing assets.
- A digest-protected V6 pilot/development/final seed plan and resumable provenance-aware evaluator.
- Versioned 12-target conditional PPO masking with action-coverage and hidden-state tests.
- An explicit scripted-HYDRA baseline plus trainable neural-HYDRA implementation and study protocol.
- A pinned Colab run that trains on development seeds only, writes rolling atomic per-episode checkpoints, restores exact RNG/optimizer state, resumes evaluation per episode, and guards the final split.

## Yellow — required scientific work

1. **Freeze code and configurations.** Tag the final code commit; store the exact environment, reward, grader, trajectory, prompt, tokenizer, model, and seed-plan versions. The 250 V5 seeds are now directly verified against the five original Drive expert-metrics files, and the frozen 1,150-seed V6 plan has zero overlap. Final code and checkpoint hashes still need to be frozen after the planned experiments.
2. **Run final paired evaluation.** Use the frozen held-out seed plan across every compared policy. The protocol proposes 200 episodes per agent–level as a starting target, with a power/precision justification recorded before results are viewed.
3. **Add uncertainty.** Report 95% confidence intervals for continuous metrics and exact binomial intervals for pass/failure rates.
4. **Run ablations.** At minimum isolate curriculum, action weighting, repair, supervisor, and security gate effects.
5. **Evaluate generalization.** Include shifted schedules, parameter perturbations, unseen names/layouts, and the preregistered scripted-versus-neural HYDRA comparison across multiple neural training seeds.
6. **Measure operational behavior.** Log raw parse validity, semantic validity, repair category, fallback/supervisor use, latency, token count, and model inference failures.
7. **Add PPO fairly or remove it from empirical comparisons.** The repository implements PPO, but the current V5 comparison artifact does not include a matched PPO row.
8. **Independent verification.** Have another person rerun extraction and at least one evaluation slice from a clean environment.

## Red — author decisions that cannot be inferred from code

- `AUTHOR-TODO`: final venue and track.
- `AUTHOR-TODO`: affiliations and institutional addresses.
- `AUTHOR-TODO`: corresponding-author email and all ORCIDs.
- `AUTHOR-TODO`: CRediT contribution allocation.
- `AUTHOR-TODO`: funding, compute credits, acknowledgements, and conflicts of interest.
- `AUTHOR-TODO`: whether anonymous-review rules apply.
- `AUTHOR-TODO`: artifact/data release location and archival DOI.
- `AUTHOR-TODO`: model checkpoint redistribution permission and license compatibility.

## Recommended paper positioning

The strongest current positioning is **environment/evaluation methodology plus an instructive negative result**, not “a solved autonomous counter-intelligence agent.” The novel story is that security-gated evaluation exposes a failure hidden by aggregate improvement, while a separately reported supervisor confirms that the environment and gate are solvable.

## Submission gate

Do not mark the paper “submission-ready” until:

```text
all AUTHOR-TODO fields resolved
AND final code/data/model hashes frozen
AND extended evaluation completed
AND planned confidence intervals reported
AND core ablations completed
AND raw/repaired/supervised results separated
AND manuscript numbers regenerated automatically
AND independent reproduction recorded
AND venue checklist passed
```
