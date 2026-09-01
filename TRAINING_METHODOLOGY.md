# Canonical Panopticon training methodology

## Research claim

The experiment tests whether a sequential LoRA SFT curriculum, trained on a
deterministic security-first expert, improves the **raw** Qwen 2.5 1.5B policy over
the immutable base model without degrading Panopticon's security requirements.
It does not test a repaired/controller-assisted policy and does not claim that
more compute alone produces correctness.

The machine-readable protocol is
[`training_specs/security_first_v5.json`](training_specs/security_first_v5.json).
That file, the source commit, and the upstream model revision define run identity.

## Experimental unit and training data

An expert episode is the data-generation unit. Fifty episodes are generated at
each of the five ordered curriculum levels. Every episode seed is derived from a
versioned training namespace; seeds are unique and disjoint from development,
canonical, and confirmation evaluation. The generator records the exact seed
list, per-episode metrics, example count, and content hashes.

The security-first expert must pass the environment's security regression tests.
Chat-formatted examples train only assistant response tokens. Canonical execution
must stop if the response boundary cannot be identified, the first batch contains
no assistant labels, or data identity/content differs on resume.

## Optimization

The five adapters are trained sequentially from easy through level 5. All LoRA,
optimizer, schedule, batching, accumulation, sequence, checkpoint, precision,
worker, and stage-RNG values are explicit in the spec. BF16 is required because a
historical FP16 workflow produced zero-loss underflow. Full determinism and the
deterministic CUDA workspace are enabled; nevertheless, GPU libraries can retain
hardware-specific numerical behavior, so hardware and package provenance is
reported rather than claiming universal bitwise identity.

Before expert generation, preflight loads the exact upstream model revision and
runs one assistant-only LoRA forward/backward/AdamW step at the canonical sequence
shape. Non-finite/zero loss or gradients, unsupported BF16, or OOM stops the run.
Each stage has an immutable metadata record; every optimizer checkpoint repeats
that identity. Resume accepts only an exact match.

After stage 5, the adapter is merged through a temporary directory and atomically
renamed. A content manifest freezes every merged-model byte before held-out
evaluation and is rechecked before every evaluation row.

## Evaluation isolation

Development, canonical, and confirmation use disjoint deterministic episode
namespaces. Canonical and confirmation each run an exact matched matrix:

- base model and candidate model;
- random, heuristic, and raw trained-policy slots;
- five levels and twenty episodes per agent/level;
- deterministic decoding, BF16, 512 prompt tokens, and 128 new tokens;
- the frozen reward/grader schemas;
- no policy repair/controller intervention and no token-truncated model turn.

Large repeated prompt/message/state bodies are replaced in canonical episode
records by SHA-256 hashes and byte counts. Actions, outputs, validation decisions,
model token telemetry, rewards, timeline metrics, final state, and grading remain
available. The environment seed, action trace, source, and hashes make the omitted
bodies reproducible and tamper-evident while keeping complete evaluation tractable.

Evaluation checkpoints one record after every episode. Final JSON is generated
only from a complete agent × level × seed matrix. The verifier independently
matches final records to sidecars and recomputes summaries from raw episodes.

## Acceptance and negative results

Integrity and scientific acceptance are separate. Integrity means identities,
lineage, completeness, schemas, finite telemetry, model bytes, plots, and hashes
are internally valid. Acceptance additionally requires, on **both** canonical and
confirmation splits:

- exact matched evaluator configuration and seed plan;
- macro grade improvement and non-worse grade at every level;
- a positive lower bound for a deterministic 5,000-sample paired-bootstrap
  interval of candidate-minus-base episode grades;
- 100% candidate pass rate at every level;
- on levels 4 and 5, security and sleepers caught not worse than base, zero
  sleepers missed, and zero false accusations.

Every check must pass. An aggregate improvement cannot override uncertainty or a
security failure. A rejected run remains useful negative evidence and must be
bundled and labelled rejected/noncanonical rather than selectively rerun, edited,
or evaluated through a different policy.

## Limits and prohibited interpretation

The fixed 20-episode held-out samples quantify only the declared environment and
seed distributions. They do not establish general safety outside Panopticon. The
expert is programmatic and can transfer its biases. SFT behavior is not equivalent
to online RL. Confirmation is a second frozen test, not permission for additional
tuning. Full expensive training is intentionally not performed by infrastructure
contributors or CI; external compute runs begin only after the version-controlled
protocol and safeguards are reviewed.
