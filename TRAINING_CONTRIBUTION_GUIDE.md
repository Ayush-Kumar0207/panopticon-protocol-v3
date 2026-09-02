# Training and reproducibility contributions

The eventual canonical workflow is designed so that a contributor supplies compute from
their own account without receiving any maintainer credential. Trustworthy
negative results are welcome; benchmark scores must never be selected or edited.

> **Current state:** development/model-selection compute is authorized by the
> committed `model_selection_v1.json` status
> `preregistered-development-compute-authorized`. External compute and
> reproducibility contributors are being recruited now.
>
> **Still sealed:** final refit, canonical evaluation, and confirmation evaluation
> remain unauthorized. The development campaign may not use either held-out
> namespace. They become eligible only after model selection finishes and a
> separate reviewed spec is committed with `status: frozen-selected-canonical`.

## Authorized development-only selection

The committed protocol authorizes one unchanged campaign command:

```bash
python tools/run_model_selection.py \
  --campaign-dir /persistent/path/panopticon-development-selection
```

The runner derives the exact 8→3→2→1 candidates, per-round data budgets,
optimization seeds and per-level seed mapping, shared development episode plans,
security gates, multi-seed aggregation, ranking, resume identities, and survivor
decisions. It can train only mechanically derived
`development-selection-candidate` specs. Those specs cannot open canonical or
confirmation. The output V6 spec is proposal-only and uses the preregistered fresh
final-refit seed; a reviewer must commit and freeze it before held-out access.

Every candidate is derived from `security_first_v5.json`, so the frozen
Ampere/native-BF16 runtime applies to development selection too: Python 3.11,
NVIDIA compute capability 8.0 or newer, at least 14 GiB VRAM, and the exact CUDA
12.1/PyTorch environment. Before the full campaign, run the repository installer,
normal validation suite, and `tools/gpu_training_probe.py`. Stop after the probe
and ask the maintainer to review its JSON evidence before spending substantial
compute.

Compatible compute may be local or cloud-based. A contributor who already owns a
suitable NVIDIA workstation or gaming laptop may not need to purchase cloud GPU
time, but “Ampere” alone is not qualification: all VRAM, native-BF16, exact-runtime,
free-disk, and numerical probes still apply. Local systems also need adequate
cooling, stable power, persistent storage, and an operating-system configuration
that can stay awake through a long resumable campaign.

## Choose the contribution type

- **Track A — environment/preflight verifier:** run the exact environment checks
  and real BF16 LoRA forward/backward/optimizer probe, then submit the qualification
  evidence. This is a small commitment and does not claim a campaign result.
- **Track B — development compute contributor:** run the authorized, frozen
  model-selection campaign and return its complete evidence chain.
- **Track C — independent artifact/reproducibility reviewer:** verify hashes,
  identities, completeness, resume lineage, gates, and survivor decisions without
  choosing results or opening held-out evaluation.
- **Track D — research contributor:** propose substantive methodology, analysis,
  or writing as a separately reviewed research contribution. Never reuse the
  canonical V5 run identity for changed methodology.

Tracks A–C receive public technical credit appropriate to accepted work but do not
carry promised authorship. Authorship, if a manuscript results, follows actual
scholarly contribution and the applicable venue/authorship standards.

## Canonical compute workflow (after a selected spec is frozen)

1. Comment on [issue #1](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/issues/1) with your GPU, VRAM, platform, persistent-storage plan, and artifact host. Wait for acknowledgement to avoid duplicate compute.
2. Fork and clone the repository, check out the exact approved commit, and create a contribution branch. Do not edit training-critical files.
3. Provision **Python 3.11 on an NVIDIA Ampere-or-newer GPU with at least 14 GiB
   VRAM**. T4/Turing is not a canonical runtime: it lacks the native BF16 profile
   frozen here, and the historical FP16 path showed loss underflow. Do not switch
   to FP16 or FP32 under this experiment identity. Run the repository installer,
   which selects the official CUDA 12.1 PyTorch wheel and fails if the resulting
   build, CUDA runtime, compute capability, or BF16 support is wrong:

   ```bash
   python tools/install_canonical_training_env.py
   ```

   Do not run `pip install -r requirements-training.txt` directly on a GPU host;
   an unspecified package index can select a CPU-only PyTorch wheel. Use your own
   cloud, Drive, GitHub, and Hugging Face accounts.
4. Put the run on persistent storage. Choose a new, empty directory; never point a new experiment at an old run.
5. Set one persistent run directory once and pass the reviewed selected spec. Then
   use exactly this five-command flow; do not add hyperparameter, seed, model,
   evaluator, or path flags:

   ```bash
   export PANOPTICON_RUN=/persistent/path/panopticon-security-first-selected
   export PANOPTICON_SPEC=training_specs/security_first_v6_selected.json

   # 1. Fail before GPU expense if source, packages, security, BF16, memory, or numerics are wrong.
   python tools/run_canonical_experiment.py --spec "$PANOPTICON_SPEC" --run-dir "$PANOPTICON_RUN" --stage preflight

   # 2. Canonical training; rerun this unchanged after interruption for automatic safe resume.
   python tools/run_canonical_experiment.py --spec "$PANOPTICON_SPEC" --run-dir "$PANOPTICON_RUN" --stage train

   # 3. Both matched canonical and confirmation base/candidate evaluations; also resumable.
   python tools/run_canonical_experiment.py --spec "$PANOPTICON_SPEC" --run-dir "$PANOPTICON_RUN" --stage evaluate

   # 4. Recompute integrity and every acceptance/security gate from raw episodes.
   python tools/run_canonical_experiment.py --spec "$PANOPTICON_SPEC" --run-dir "$PANOPTICON_RUN" --stage verify

   # 5. Create a bounded evidence ZIP plus a hash index for separately hosted large artifacts.
   python tools/build_submission_bundle.py "$PANOPTICON_RUN" --spec "$PANOPTICON_SPEC"
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
   security gates before writing checksums. Large repeated prompt/message/state
   fields are stored once as deterministic gzip, content-addressed blobs under
   `raw_evidence/`; verification decompresses and re-hashes **every** referenced
   blob. The bounded Git-facing ZIP can externalize these files, but its manifest
   keeps their byte identities mandatory.
   Exit code `2` means evidence is intact but scientifically rejected; it is not success.
9. Upload large model/checkpoint artifacts using your account. Do not commit model
   archives, checkpoints, `.env` files, cloud credential files, or tokens.
10. Open a PR using the training-reproduction template. Link the issue, model,
    immutable logs/results/provenance/checksums, and disclose every interruption or deviation.

## What must remain immutable

The reviewed selected spec is the source of truth. V5 currently defines the
reproducible method baseline but is not authorized as the final experiment. The
run fingerprint binds the selected spec's complete content to the source commit.
The frozen runtime profile is
`ampere-bf16-cu121`: Python 3.11, `torch==2.2.1+cu121`, CUDA runtime 12.1,
native BF16, compute capability 8.0 or newer, and at least 14 GiB VRAM. A T4 is
not accepted. FP32-on-T4 and stabilized/scaled-FP16-on-T4 remain unvalidated
research proposals and require separate preregistered identities plus numerical
and outcome-equivalence evidence before they can be offered to contributors.

The low-VRAM profile is training-critical because sequence length, epochs, batch
size, accumulation, checkpointing, and precision affect the optimization path.
Changing to the old CPU/basic or default-GPU convenience profiles is a different
experiment, not a valid resume.

Read [`KNOWN_TRAINING_FAILURE_MODES.md`](KNOWN_TRAINING_FAILURE_MODES.md) before
offering compute. It explains why older completed/high-scoring artifacts are not
canonical and maps every recoverable failure to the current fail-closed safeguard.
Read [`TRAINING_METHODOLOGY.md`](TRAINING_METHODOLOGY.md) for the experimental
claim, unit of analysis, paired statistics, evaluation isolation, and limitations.
Read [`MODEL_SELECTION_PROTOCOL.md`](MODEL_SELECTION_PROTOCOL.md) for the bounded
search, mechanical runner, multi-seed no-shopping rule, hard eligibility gates,
and the rule that held-out data stays sealed.

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
