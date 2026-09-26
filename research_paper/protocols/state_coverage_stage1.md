# Stage 1 State-Coverage Protocol

**Status:** Proposed choices for maintainer review (Issue #1).

## 1. Research Aims and Estimands
*   **H1 (Coverage):** As learner episodes progress, the learner encounters observations with increased distance from the *exact training-demonstration* reference, potentially particularly after behavioral divergence.
*   **H2 (Disagreement):** Among turns with valid raw model actions and validated expert labels, action disagreement is associated with lower training-support coverage.
*   **H3 (Later Safety Outcomes):** At the episode level, characterize exploratory associations between early coverage/divergence diagnostics and later endpoints: missed sleepers, false accusations, final security, and gate failure.
*   **Primary Unit:** Episode/environment seed.

## 2. Data Sources
*   **Reference T (Training Support):** The immutable, versioned training-demo corpus that produced the model checkpoint. This is the primary reference.
*   **Reference E (Behavioral Expert):** Independently verified same-seed expert development trajectories. **Note: Absent by default in this Stage 1 PR.**

## 3. Observation Allowlist
To compute coverage, we propose a compact intersection of recoverable numeric and categorical summaries:
*   `turn / max_turns` and phase category.
*   Worker count, unique department count, and count of worker rows with visibly high suspicion (e.g., `>0.50`).
*   Visible active leak count, visible canary-match count, and triggered canary count.
*   Visibly rounded security and revenue; active double-agent count.
*   Observable legal-action signature derived from `models.validate_action`.

**Explicit Exclusions:** `Worker.hidden_state`, `is_sleeper`, `generation`, `cover_integrity`, `leak_cooldown`, `activation_turn`, `false_flag_target`, `dead_switch_armed`, `EnvironmentState`, `hydra_memory`, and unverified source attribution.

## 4. Novelty and Calibration Proposal
*   **Stratification:** Estimate reference neighborhoods within the same game level (if enough Reference T samples exist).
*   **Neighbors:** k = 5 independent eligible expert episode/seed groups. For each group, use its closest eligible reference turn, then average the five smallest group distances.
*   **Distance:** A mixed-type arithmetic mean with equal weight per included feature. Each numeric feature contributes `min(5, abs(query - reference) / expert_reference_IQR)`, with zero IQR replaced by 1. Numeric clipping applies to the pairwise standardized difference, not to separately clipped coordinates. Each categorical feature contributes 0 when equal and 1 when unequal. Fit scaling using eligible reference turns only.
*   **Calibration:** Within the same game level, leave one expert episode/seed group out. Score its turns against the other independent groups, then take the median turn novelty within the held-out episode. Every eligible episode contributes one calibration value.
*   **Threshold:** The proposed threshold is the 95th percentile of held-out expert episode medians. Compare it only against the median of a learner episode's valid turn novelty scores, using a strict `>` comparison. Individual turn scores remain diagnostics and are not compared directly against this episode-level threshold. Missing or incomparable turns make the proposed complete-episode score unavailable.

## 5. Action Canonicalization
*   **Canonical Tuple:** `(action_type, target, sub_action)`.
*   Omitted optional sub-actions are `"none"`, and absent targets are `""`.
*   Pre-declared turn categories: `raw_parse_failure`, `raw_semantic_invalid`, `raw_valid_expert_label_unavailable`, `raw_valid_expert_label_invalid`, `raw_valid_agreement`, `raw_valid_disagreement`, and `executed_action_different_from_raw`.

## 6. Training-Reference Provenance

Reference T must be the exact, immutable training-demonstration
corpus associated with the evaluated checkpoint. A newly generated
expert corpus is not an acceptable substitute for the primary
training-support analysis.

Upstream training code writes formatted observation text and action
JSON. Episode and seed information must therefore be recovered from
verified training metadata or another trustworthy original record;
it must not be inferred from row ordering alone.

Before analyzing real artifacts, verify and record:

- The exact checkpoint and training-stage identity.
- The original training dataset paths and SHA-256 hashes.
- The training metadata and source-code revision.
- The episode-to-seed mapping, if independently recoverable.
- Whether the dataset preserves enough information for the proposed
  observation features and episode-held-out calibration.

If the original corpus, feature values, or independent episode
identities cannot be verified, report the corresponding analysis
as unavailable. Do not silently substitute regenerated trajectories.

Reference E, consisting of verified independent expert development
trajectories, remains absent by default.

## 7. Feature Recoverability and Leakage Controls

The primary coverage feature set must be a versioned intersection
of information recoverable from both Reference T and the learner's
recorded observation_before.

Check proposed features against the actual training text generated
by train_trl_v2.format_observation. A feature is eligible only when
its extraction can be demonstrated with synthetic parity fixtures
and its semantics agree across both sources.

In particular, rounded values, displayed suspicion percentages,
canary summaries, and observation-derived action legality require
explicit recoverability checks before inclusion.

Do not use hidden worker attributes, EnvironmentState, HYDRA state,
ground-truth sleeper identities, unverified attribution, or
information appearing only after the learner's current decision.

If the required common feature set is unavailable, do not assign
a zero novelty score or claim that training coverage was measured.

## 8. Reproducibility and Exclusion Reporting

A future real-data analysis must emit a versioned manifest recording
the source revision, checkpoint identity, training-reference hashes,
feature schema and extractor revision, eligible episode groups,
game-level stratification, k, calibration percentile, and exclusions.

The proposed defaults are k = 5 independent reference episodes
and a 95th-percentile expert held-out novelty threshold.

Validate episode identity and continuity, required observation
evidence, and actual content-addressed evidence bytes before
including an episode. Record missing or unverifiable evidence
as an explicit exclusion reason.

Keep synthetic test results separate from real research findings.
The present implementation and fixtures demonstrate proposed
methods; they do not establish empirical H1, H2, or H3 results.

## 9. Questions for Maintainer Review

Before any real development-artifact analysis or new rollouts,
request confirmation of:

1. Availability and identity of the exact training corpus.
2. Availability of trustworthy per-episode training seed mapping.
3. The approved, mutually recoverable observation feature schema.
4. The exclusion policy if training provenance or feature
   recoverability is incomplete.

Proceed to real-artifact work only after the relevant approval.
