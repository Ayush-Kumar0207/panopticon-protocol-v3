# Dataset Card — Panopticon Security-Gated Expert Trajectories V5

## Summary

The dataset consists of synthetic state-observation/action examples produced by a deterministic security-first expert interacting with the Panopticon environment. It is intended for supervised imitation of structured defensive actions, not for learning from real employee, intelligence, or cybersecurity records.

## Current checked-in summary

| Level | Accepted expert episodes | Weighted examples | Max tokens |
|---|---:|---:|---:|
| Easy | 50 | 7,430 | 512 |
| Medium | 50 | 13,166 | 512 |
| Hard | 50 | 18,414 | 512 |
| Level 4 | 50 | 23,889 | 512 |
| Level 5 | 50 | 25,997 | 512 |
| **Total** | **250** | **88,896** | — |

Weighted examples are not 88,896 independent episodes. Strategically important or rare actions can appear multiple times by configured duplication.

## Generation and acceptance

For each seeded episode, the expert reads the agent-visible observation and selects an `AgentAction`. After completion, generation fails closed unless:

- final security is at least 90;
- every spawned sleeper is caught;
- no sleeper is missed; and
- there are no false accusations.

Accepted observation/action pairs are formatted through the base tokenizer's chat template. The training loss is applied to completion tokens rather than prompt tokens.

## Data fields

The final archive should contain an immutable trajectory ID, level, episode seed, turn, redacted observation, expert action, action weight, reward/grader/trajectory schema versions, source commit, tokenizer revision, max sequence length, truncation metadata, and gate result.

## Privacy and sensitive data

Worker names are synthetic codenames and game events are simulated. No personal data is intentionally included. Before release, scan prompts/logs for accidentally copied credentials, local paths, tokens, emails, or model-provider responses.

## Intended uses

- structured action imitation inside the matching Panopticon schema;
- curriculum and class-imbalance research;
- studies of compounding error and intervention provenance; and
- reproducible benchmark baselines.

## Out-of-scope uses

- assessing real employees, insider-threat risk, loyalty, or criminality;
- training operational surveillance or disciplinary systems;
- claiming real-world counter-intelligence competence; and
- reuse under a changed environment/action schema without regeneration and validation.

## Biases and limitations

- Expert demonstrations encode one hand-authored policy's priorities.
- Fail-closed selection removes unsuccessful trajectories, narrowing state coverage.
- Duplication changes action frequency without adding independent state diversity.
- Expert-only trajectories underrepresent states caused by learner errors.
- Fictional terminology may carry cultural or historical associations.
- The current summary alone does not prove final evaluation seed disjointness; the release must publish or commit to auditable split hashes.

## Recommended improvements

Archive unweighted trajectories; add DAgger-style learner-state labels; include safe counterexamples and preference pairs; quantify duplication; publish split/seed hashes; and document exact chat-template/truncation behavior.
