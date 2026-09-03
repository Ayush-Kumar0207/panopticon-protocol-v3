# Contributor onboarding: development training and reproducibility

This is the shortest safe path from interest to an accepted Panopticon
contribution. The current authorization covers only the frozen
**development/model-selection campaign**. Final refit and canonical and
confirmation evaluation remain sealed.

## 1. Choose one bounded track

| Track | Suitable for | Completion evidence |
|---|---|---|
| A — preflight | A contributor checking whether a machine is suitable | Exact environment report, test output, and the real BF16 LoRA numeric probe |
| B — development compute | A contributor with a stable compatible NVIDIA machine and persistent storage | The complete, unedited 8→3→2→1 campaign directory |
| C — independent review | A contributor reviewing an existing campaign without training | Completed review checklist, recomputed hashes, and discrepancies |
| D — research | Methodology, analysis, interpretation, or writing beyond execution | A separately reviewed proposal, analysis, or PR; changed methods require a new experiment identity |

Tracks A–C can receive public technical attribution when accepted. Authorship is
never promised for compute alone; it depends on a substantial scholarly
contribution if a manuscript results.

## 2. Claim work before spending compute

Comment on [intake issue #1](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/issues/1)
and wait for a public acknowledgement. Include only:

- the desired track;
- GPU model, VRAM, compute capability if known, CPU, RAM, OS, and Python version;
- approximate uninterrupted availability and expected interruption pattern;
- persistent-storage plan and approximate free capacity;
- intended artifact host and retention period; and
- the exact commit you intend to run.

Do not publish email addresses, phone numbers, tokens, account identifiers,
private paths, cloud-console screenshots, or credentials. Use your own accounts;
maintainer credentials are never needed.

## 3. Qualify the machine before expensive work

Panopticon requires Python 3.11, NVIDIA compute capability 8.0 or newer, native
BF16, at least 14 GiB VRAM, and the exact `torch==2.2.1+cu121` build. A T4 or a
typical 6–8 GB laptop GPU does not qualify. A compatible 16 GB workstation or
gaming laptop may run locally, subject to thermals, storage, and stability.

From a clean checkout of the acknowledged commit:

```bash
python tools/install_canonical_training_env.py
python -m pip check
python tools/training_preflight.py --spec training_specs/security_first_v5.json --output /persistent/path/panopticon-preflight.json
```

The preflight invokes the real one-step LoRA forward/backward/optimizer probe.
A diagnostic-only result is expected for the provisional baseline and is not
permission to train that baseline. Stop on any error; do not switch precision,
package versions, model revision, sequence length, seeds, or thresholds to make
the check pass.

## 4. Run only the authorized campaign

Track B uses a new empty directory on persistent storage outside the source
checkout:

```bash
python tools/run_model_selection.py --campaign-dir /persistent/path/panopticon-development-selection
```

Rerun that exact command after an interruption. The runner locks the source and
protocol identities, validates existing evidence, and resumes compatible work.
Never edit the run lock or campaign state, reuse another experiment directory,
delete an unfavorable record, or add flags that alter candidates, seeds, budgets,
ranking, or evaluation.

The runner executes the preregistered topology: eight candidates with one seed,
three survivors with two seeds, two survivors with three seeds, then one
mechanical proposal. That is 20 candidate-seed runs. It may use only training and
development namespaces. Do not attempt canonical or confirmation evaluation.

## 5. Preserve every deliverable

Submit the complete directory, including unsuccessful and interrupted attempts.
At minimum, reviewers must be able to locate and verify:

- `campaign_lock.json` and `campaign_state.json`;
- every generated candidate spec and its exact source/spec/model identities;
- preflight reports and dependency/runtime information;
- training logs, optimizer diagnostics, checkpoints, expert-data identities,
  interruption and resume events, and failure records;
- every raw development episode and its content-addressed evidence;
- per-seed and aggregate metrics, safety/eligibility results, and bootstrap data;
- each round decision and survivor list;
- `selection_decision.json` and the proposed
  `security_first_v6_selected.json`; and
- a recursive file inventory with byte sizes and SHA-256 hashes.

Do not rename, prune, rewrite, or manually “repair” runner outputs. A negative or
incomplete run is useful evidence when its status and failure are preserved.

## 6. Submission without large files in Git

1. Keep checkpoints, model weights, raw evidence, archives, and credentials out
   of Git.
2. Upload the untouched campaign directory or a lossless archive to a durable
   artifact host under your own account. Prefer an immutable/versioned link.
3. Open a PR containing only a concise Markdown report and any small
   machine-readable decisions/manifests appropriate for review.
4. Link issue #1, the acknowledged commit, the artifact URL, the archive hash,
   the recursive SHA-256 manifest, and the exact reproduction command.
5. Disclose every interruption, warning, deviation, missing file, and retention
   limit. Never put a token or private path in the PR or logs.

The proposed selected spec is review material only. A maintainer must review and
commit a new spec with status `frozen-selected-canonical` before final refit or
either held-out split can run.

## 7. Acceptance checklist

A reviewer should mark each item explicitly:

- [ ] The claimed commit is clean and matches `campaign_lock.json`.
- [ ] Python, CUDA, PyTorch, GPU capability, BF16, VRAM, model revision, and all
      training-critical dependencies match the frozen requirements.
- [ ] The real LoRA probe passed with finite nonzero loss and gradients and
      recorded peak allocation.
- [ ] All 20 registered candidate-seed runs are present, or the campaign is
      clearly labelled incomplete with its failure evidence.
- [ ] Candidate values, seeds, budgets, development episodes, gates, ranking,
      and 8→3→2→1 decisions match `model_selection_v1.json` exactly.
- [ ] Every registered seed is included; no seed shopping, manual override,
      deleted failure, or favorable rerun occurred.
- [ ] No canonical or confirmation namespace or result was accessed.
- [ ] Raw evidence, summaries, decisions, hashes, and provenance agree.
- [ ] Every security gate and every round eligibility condition was recomputed
      from the submitted evidence.
- [ ] The external archive hash and recursive manifest verify.
- [ ] The outcome is labelled correctly using the definitions below.

### Status vocabulary

- **Campaign complete:** every registered development run and mechanical decision
  exists. This alone is not a successful policy claim.
- **Evidence valid:** identities, completeness, provenance, hashes, and
  calculations verify. Valid evidence may still be negative.
- **Development winner proposed:** the mechanical development rule selected one
  configuration. It is not yet a frozen final model.
- **Scientifically accepted policy:** reserved for a later frozen refit whose
  canonical and confirmation evidence both pass every integrity, uncertainty,
  quality, and security gate. No current development run may use this label.

## 8. Safety boundary

Use only the repository's synthetic, authorized environment. Do not connect the
agent to real systems, accounts, networks, targets, production data, or private
datasets. Do not weaken approval gates, logging, deterministic identities, seed
separation, or evaluation isolation. Stop and report ambiguity before spending
additional compute.

For the detailed experimental rules, read
[`TRAINING_CONTRIBUTION_GUIDE.md`](TRAINING_CONTRIBUTION_GUIDE.md),
[`MODEL_SELECTION_PROTOCOL.md`](MODEL_SELECTION_PROTOCOL.md), and
[`KNOWN_TRAINING_FAILURE_MODES.md`](KNOWN_TRAINING_FAILURE_MODES.md).
