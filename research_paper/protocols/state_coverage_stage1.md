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
*   **Neighbors:** k = 5 distinct eligible expert reference observations.
*   **Distance:** A mixed-type Gower-style average. Numeric differences are standardized (`(x - expert_median) / expert_IQR`) and clipped `[-5, 5]`. Categorical differences are `0` if equal, `1` if unequal.
*   **Calibration:** Leave one expert episode/seed group out during fitting to prevent a query state from being its own neighbor.
*   **Threshold:** The proposed low-coverage threshold is the 95th percentile of expert held-out novelty scores.

## 5. Action Canonicalization
*   **Canonical Tuple:** `(action_type, target, sub_action)`.
*   Omitted optional sub-actions are `"none"`, and absent targets are `""`.
*   Pre-declared turn categories: `raw_parse_failure`, `raw_semantic_invalid`, `raw_valid_expert_label_unavailable`, `raw_valid_expert_label_invalid`, `raw_valid_agreement`, `raw_valid_disagreement`, and `executed_action_different_from_raw`.
