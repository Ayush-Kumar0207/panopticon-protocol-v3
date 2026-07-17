# Limitations and Threats to Validity

## Construct validity

- “Security,” “intelligence,” and “adaptability” are hand-designed simulation metrics, not externally validated constructs.
- Catching fictional sleepers is not equivalent to detecting real insider threats or AI attacks.
- A five-dimensional composite embeds normative weights.
- The hard gate is clearer than the composite but still reflects authored thresholds.

## Internal validity

- The expert, environment, reward, and grader share assumptions and may share bugs.
- Repair/supervisor behavior can be incorrectly attributed to the model without detailed logs.
- Training and final seed separation must be proven, not assumed from a single seed label.
- Stochastic decoding and environment randomness require paired control.
- Some stored configuration fields do not affect transitions.

## Statistical conclusion validity

- The current 20 episodes per policy–level provide limited rare-failure evidence.
- Existing summaries lack final paired confidence intervals and hypothesis tests.
- Five level means may hide within-level multimodality or rare catastrophic seeds.
- Checkpoint/experiment selection history may introduce researcher degrees of freedom.

## External validity

- Reported results use the scripted-memory adversary. A neural HYDRA is implemented but has no completed multi-seed held-out comparison and is not a model of human attackers.
- Actions and evidence are specific to the fictional domain.
- No current result demonstrates performance on real agent applications, RAG systems, or tool APIs.
- One 1.5B model family does not establish generality across model scales/providers.

## Engineering threats

- The new 12-target action schema covers the current worker capacity, but it invalidates legacy eight-target PPO checkpoints unless they are explicitly migrated and retrained.
- Joint masks are now applied and tested, but no matched PPO benchmark has yet been trained under the new schema.
- Container module paths are now aligned, but no clean Docker build was possible in this workspace; the lightweight image also omits local neural-inference dependencies.
- The demo server is single-session, unauthenticated, and non-durable. Privileged state/render routes are disabled by default but still lack RBAC when explicitly enabled.
- Persisting only public Pydantic state is insufficient for exact RNG/private-engine replay.

## Mitigations planned

Independent invariant/property tests; frozen schemas and hashes; larger paired evaluation; explicit raw/repair/supervisor telemetry; ablations; OOD variants; multiple model families and training seeds; external target adapters using authorized sandboxes; and independent reproduction.

## Claim boundary

The paper may claim a research environment, evaluation method, preliminary case study, and implemented benchmark extension. It must not claim production readiness, regulatory compliance, real-world detection efficacy, general safety, or a commercially validated market outcome.
