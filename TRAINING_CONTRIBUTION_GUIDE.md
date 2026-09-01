# Training and reproducibility contributions

The canonical workflow is designed so that a contributor supplies compute from
their own account without receiving any maintainer credential. Trustworthy
negative results are welcome; benchmark scores must never be selected or edited.

## Choose the contribution type

- **Compute-only reproduction:** run the already-frozen experiment and return its complete evidence chain.
- **Code contribution:** improve tests, safety, documentation, or infrastructure without claiming a new canonical result.
- **Research/methodology change:** propose a new spec and validation protocol. Never reuse the canonical V5 run identity.

## Canonical compute workflow

1. Comment on [issue #1](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/issues/1) with your GPU, VRAM, platform, persistent-storage plan, and artifact host. Wait for acknowledgement to avoid duplicate compute.
2. Fork and clone the repository, check out the exact approved commit, and create a contribution branch. Do not edit training-critical files.
3. Create a Python 3.11 or 3.12 environment and run `python -m pip install -r requirements-training.txt`. Use your own Colab, cloud, Drive, GitHub, and Hugging Face accounts.
4. Put the run on persistent storage. Choose a new, empty directory; never point a new experiment at an old run.
5. Set one persistent run directory once. Then use exactly this five-command flow;
   do not add hyperparameter, seed, model, evaluator, or path flags:

   ```bash
   export PANOPTICON_RUN=/persistent/path/panopticon-security-first-v5

   # 1. Fail before GPU expense if source, packages, security, BF16, memory, or numerics are wrong.
   python tools/run_canonical_experiment.py --run-dir "$PANOPTICON_RUN" --stage preflight

   # 2. Canonical training; rerun this unchanged after interruption for automatic safe resume.
   python tools/run_canonical_experiment.py --run-dir "$PANOPTICON_RUN" --stage train

   # 3. Both matched canonical and confirmation base/candidate evaluations; also resumable.
   python tools/run_canonical_experiment.py --run-dir "$PANOPTICON_RUN" --stage evaluate

   # 4. Recompute integrity and every acceptance/security gate from raw episodes.
   python tools/run_canonical_experiment.py --run-dir "$PANOPTICON_RUN" --stage verify

   # 5. Create a bounded evidence ZIP plus a hash index for separately hosted large artifacts.
   python tools/build_submission_bundle.py "$PANOPTICON_RUN"
   ```

   The training command intentionally repeats preflight before creating the run
   lock. This guards against a changed checkout/runtime between commands. `--stage
   all` remains available for automation, but the split commands make expensive
   progress and failures easier for a beginner to inspect.

6. After a disconnect, restore the same repository commit, dependency environment,
   and run directory, then rerun the same command. A compatible checkpoint resumes;
   a changed seed, spec, source, model, or trainer configuration fails without deleting data.
7. Evaluation performs matched base/candidate canonical evaluation from root seed
   42, then matched base/candidate confirmation evaluation from root seed 43.
   Deterministic split-specific namespaces make the actual episode seeds disjoint
   from training and development. Do not use either held-out result to change training.
8. Verification checks checkpoint lineage, expert-data hashes and exact seeds,
   finite optimizer evidence, merged safetensors headers, tokenizer/config files,
   curriculum completion, every expected episode and sidecar, raw-summary equality,
   evaluator isolation, paired-bootstrap improvement uncertainty, plots, and all
   security gates before writing checksums.
   Exit code `2` means evidence is intact but scientifically rejected; it is not success.
9. Upload large model/checkpoint artifacts using your account. Do not commit model
   archives, checkpoints, `.env` files, cloud credential files, or tokens.
10. Open a PR using the training-reproduction template. Link the issue, model,
    immutable logs/results/provenance/checksums, and disclose every interruption or deviation.

## What must remain immutable

`training_specs/security_first_v5.json` is the source of truth. The run fingerprint
binds its complete content to the source commit. GPU names are recorded but not
part of experiment identity: a T4 is not uniquely required. Any CUDA GPU with the
required memory can run the same fixed low-VRAM profile. Optimizer checkpoints are
portable when the exact software/configuration is restored, although floating-point
kernels can still produce small hardware-dependent differences.

The low-VRAM profile is training-critical because sequence length, epochs, batch
size, accumulation, checkpointing, and precision affect the optimization path.
Changing to the old CPU/basic or default-GPU convenience profiles is a different
experiment, not a valid resume.

Read [`KNOWN_TRAINING_FAILURE_MODES.md`](KNOWN_TRAINING_FAILURE_MODES.md) before
offering compute. It explains why older completed/high-scoring artifacts are not
canonical and maps every recoverable failure to the current fail-closed safeguard.
Read [`TRAINING_METHODOLOGY.md`](TRAINING_METHODOLOGY.md) for the experimental
claim, unit of analysis, paired statistics, evaluation isolation, and limitations.

## Held-out data rule

- Training trajectories derive only from the training namespace and root seed 42.
- Development/validation derives from its disjoint namespace and root seed 41; it is not release evidence.
- Canonical held-out evaluation derives from its disjoint namespace and root seed 42 once the model is frozen.
- Confirmation derives from its disjoint namespace and root seed 43 on the same frozen model and writes separate files.

Never tune on canonical or confirmation results, rerun individual unfavorable
episodes, delete JSONL lines, or replace an output with a favorable run.

## Credit

Accepted work retains PR/commit attribution. Substantive accepted reproductions
can be named in a **Training & Reproducibility Contributors** section and linked
beside their external artifacts. Research credit reflects the actual contribution;
there is no payment, employment implication, or promised GitHub badge.
