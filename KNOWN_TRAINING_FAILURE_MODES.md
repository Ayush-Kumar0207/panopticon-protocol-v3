# Known Panopticon training failure modes

This is the forensic status of training evidence retained in the repository and
reachable Git history. It is deliberately conservative: **none of the historical
items below is canonical acceptance evidence**. The machine-readable inventory is
[`training_specs/historical_attempts.json`](training_specs/historical_attempts.json).
External Colab/Drive state was not treated as evidence unless it was transcribed
into a committed artifact. `unknown / insufficient evidence` is not a guess.

## Historical findings

| Attempt | What completed | Failure or limitation | Classification |
|---|---|---|---|
| April loss-zero debug series (`bc53d37`, `d282b3c`, `bbc5a7b`) | Troubleshooting commits | Historical zero loss; `bbc5a7b` explicitly identifies FP16 underflow, while other causal contributions cannot be reconstructed | failed/diagnostic |
| `panopticon-ep50-v2` invocations 1–2 | Easy expert data | Both OOM at optimizer step 0 requesting 898 MiB with about 102 MiB free on a 14.74 GiB GPU | failed |
| `panopticon-ep50-v2` invocation 3 | All five stages and merged model | Source commit, upstream model revision, config seed, full evaluation, and acceptance chain are missing | training-complete, noncanonical |
| `panopticon-ep50-v2` later invocations | 12 completed-state reuses/re-merges | Completion messages are not independent reproductions | non-runs |
| `evaluation_snapshot_apr26.json` | 2 episodes per agent/level | Too small and unbound to an immutable model; trained policy caught no sleepers and reached security 0.3 on hard/advanced levels | preliminary failed evidence |
| `fixed-v3-ep20` event stream | Final merge after 16 invocations | Fourteen invocations lack `run_complete`; source/profile/data schemas changed inside one logical directory; most interruption causes were not recorded | mixed-history, noncanonical |
| Security-First V5 raw model | Full transcribed 20-episode/agent/level comparison | Macro grade improved by 0.060517, yet nine advanced-level acceptance/security checks failed | evaluated and rejected |
| Security-First V5 supervisor | Controller passed transcribed gates | It is a deterministic controller, not evidence that the raw trained model passed | separate system only |
| V6 pilot r0–r3 | Pilot iterations | NumPy serialization failure, dependency coupling, then a corrected 512/128/no-truncation context contract invalidated earlier model rows | experimental/provisional |
| Training-safety CI runs 1–3 (2026-09-01) | Checkout/static validation only; no training | Runs 1–2 omitted the declared PyTorch test dependency. Run 3 then proved the prior dependency set was unresolvable: unbounded `fastmcp` resolution required Pydantic 2.11.7+ while the lock required 2.6.1. All later gates were skipped. | failed infrastructure, no training evidence |

Other source-history warnings are also material: assistant labels were repaired in
`428cd41`; hidden training failures and shell logging were repaired in `a80b76a`
and `67117b7`; evaluation was redirected to the persistent merged model in
`83e5b97`; prompt overflow handling changed in `9669cd`. The exact affected run
set is `unknown / insufficient evidence`, so the safeguards cover every new run.

## Safeguards now required

| ID | Automated prevention/detection |
|---|---|
| S01 | A clean source commit, complete canonical spec, exact upstream model revision, and training-critical values form one immutable run fingerprint. Preflight stops on dirty/stale source. |
| S02 | Canonical BF16 is fixed to Python 3.11 and the official `torch==2.2.1+cu121` build on Ampere-or-newer hardware. The installer and preflight reject CPU-only torch, a wrong CUDA build/runtime, T4/Turing, unsupported BF16, or a changed precision. Before expert generation, a real one-step LoRA forward/backward/optimizer probe rejects OOM, non-finite/zero loss, or non-finite/zero gradients. FP32/FP16 alternatives require new preregistered identities. |
| S03 | VRAM is checked before work and the actual model probe measures peak allocation before expensive data generation or training. |
| S04 | Run, stage, checkpoint, data, merged-model, and evaluation metadata must share the fingerprint/source/spec lineage. Mismatched resume fails closed; incomplete merges are quarantined, never silently overwritten; exact merged-model bytes are frozen and rechecked before every evaluation row. |
| S05 | Optimizer checkpoints are automatic and retained per the spec. A rerun resumes only a compatible checkpoint; interruption does not imply completion. |
| S06 | Assistant-label token counts, finite optimizer telemetry, positive finite final loss, exact expert seed list, data counts, and data/metrics hashes are verified for every level. |
| S07 | Training, development, canonical, and confirmation seeds occupy deterministic non-overlapping namespaces. Canonical evaluation fixes the raw-model policy, BF16, 512 prompt tokens, 128 new tokens, deterministic decoding, zero interventions, and zero token-truncated turns. |
| S08 | Completion requires the exact agent × level × episode matrix in final JSON and episode sidecars, matching progress, plots, all lifecycle events, model headers, and content hashes. A partial 17/20 evaluation receives `STOP`, never “complete.” |
| S09 | Acceptance is recomputed from raw episode summaries for both canonical and confirmation splits. Every gate and a deterministic paired-bootstrap positive-improvement check must pass; aggregate grade cannot override uncertainty or any security failure. Raw model and supervisor/controller identities cannot be substituted. |
| S10 | Contributor-facing entry points convert expected failures to `STOP:` messages; the verifier distinguishes artifact integrity from scientific acceptance and returns a failing exit code for rejected evidence. |
| S11 | CI installs the complete version-controlled training requirements, runs `pip check`, and then executes every test/security gate. Canonical training dependencies are isolated from the legacy server/deployment chain, and a regression test rejects ranges or drift between the experiment spec and training requirements before contributor GPU work. The research PR does not modernize the server/Docker dependency set. |
| S12 | Canonical episode JSON keeps compact hashes, while every omitted prompt, message list, and before/after observation is preserved once as deterministic gzip in a content-addressed `raw_evidence/` tree. Verification decompresses, canonicalizes, and re-hashes every reference; missing, corrupt, substituted, path-traversing, or noncanonical blobs fail. Large blobs may stay outside Git, but remain mandatory in the hashed artifact manifest. |
| S13 | Model selection is a locked, resume-aware 8→3→2→1 campaign. Real execution requires an explicit committed authorization state; candidate specs are derived exactly from the protocol and can access only training/development. Eligibility requires every registered optimization seed, metrics aggregate without seed shopping, altered evidence/identity fails, and the fresh-seed V6 output remains proposal-only until reviewed and committed. |

The safeguards intentionally do not change historical thresholds, graders, reward
schemas, or security tests. More epochs, a larger model, or extra GPU are not a
remedy for identity, isolation, integrity, or acceptance failures.

## Canonical interpretation rule

A run may be called accepted only when both canonical and confirmation evaluation
bundles are complete, integrity-verified, and every recomputed acceptance check
passes. A training-complete or high-scoring run that fails any gate remains valid
negative evidence and must be labelled **rejected/noncanonical**, not repaired by
deleting episodes, changing thresholds, or substituting a controller.
