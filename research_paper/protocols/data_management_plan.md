# Data Management and Archival Plan

## Artifact classes

| Class | Examples | Retention | Public release |
|---|---|---|---|
| Source/config | code, prompts, schemas, seed-plan hash | permanent | yes, subject to review |
| Synthetic training data | redacted observations/actions | permanent versioned release | yes if licenses permit |
| Evaluation logs | per-turn raw output, actions, outcomes | permanent research archive | redacted/controlled as needed |
| Model artifacts | LoRA adapter, merged checkpoint | permanent identity manifest | license-dependent |
| Customer/pilot evidence | prompts, tool traces, findings | contract-defined minimum | no by default |
| Generated paper assets | tables, plots, analysis report | permanent | yes |

## Naming and versioning

Use immutable experiment IDs containing date, code commit prefix, model revision prefix, reward/grader schema, and seed-plan hash prefix. Never overwrite a completed experiment directory.

## Integrity

- SHA-256 manifests for code snapshots, datasets, models, logs, and final outputs.
- Append-only per-episode logs with atomic checkpoints.
- Signed release tag and archived DOI where possible.
- A machine-readable manifest connecting every table/figure to input hashes.

## Privacy and security

The simulation uses synthetic data. External-target pilots must redact secrets, encrypt evidence at rest/in transit, restrict access by tenant/project role, define deletion deadlines, and avoid storing full responses when hashes/signals suffice. Incident response is required for unexpected personal or confidential data.

## Availability

At submission, publish the code and compact synthetic artifacts or provide an anonymous archive. Large logs/checkpoints may use an archival repository with documented access conditions. Record any unavailable artifact and reason.

## Ownership

**AUTHOR-TODO:** identify data steward, repository owner, archival service, DOI, retention period, customer-data controller/processor roles, and model-license obligations.
