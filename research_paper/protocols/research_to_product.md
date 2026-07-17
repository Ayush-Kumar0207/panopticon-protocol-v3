# Research-to-Product Strategy

## Product thesis

The nearest credible product is a **long-horizon agentic-AI evaluation and red-team platform**, not a production firewall. Panopticon's differentiator is temporal: delayed activation, apparently benign behavior, adaptive avoidance, multi-step evidence, and utility-versus-security constraints.

## Wedge product

> A long-horizon adversarial evaluation platform for agentic AI systems that measures prompt/tool attacks, delayed compromise, intervention dependence, and utility preservation across multi-turn campaigns.

## Stage 1 — Research benchmark and local SDK

- Versioned scenario DSL and target-adapter protocol.
- Mock/sandbox tools and synthetic secrets.
- Append-only event/provenance logs.
- Scripted adversary packs as the stable baseline; learned campaign selection only as an explicitly labeled experimental stressor.
- Constraint Masking Gap, intervention-dependence, time-to-compromise, attack-success, utility-preservation, and false-positive metrics.
- OWASP/MITRE/NIST coverage crosswalk with version pins.
- CI exit codes and JSON/SARIF-like report export.

**Buyer/user:** AI platform, model risk, and red-team engineers.

## Stage 2 — Pilot evaluation service

- Authenticated tenant/project/campaign APIs.
- OpenAI-compatible, Bedrock, Azure, local, LangChain/LlamaIndex, MCP, RAG, and custom HTTP adapters.
- Encrypted evidence, retention policy, secret redaction, and customer-managed keys where needed.
- Scheduled regression suites and release gates.
- Jira/Slack/SIEM/webhook outputs.
- Reproducible signed reports with target/model/config hashes.

**Evidence required:** benchmark against established scanners on temporal/multi-turn attacks, precision/recall for deterministic detectors, customer pilot findings, latency/cost, and repeatability.

## Stage 3 — Runtime gateway

Only after enough evaluation evidence:

- session-scoped risk state;
- prompt/tool/output policy enforcement;
- DLP and secret detection;
- approval workflows;
- low-latency fallbacks;
- SSO/SAML, RBAC, immutable audit, tenant isolation, and SOC 2 program.

## Stage 4 — Agentic SOC layer

Correlate intent across users, agents, retrieval, tools, and time. Use evaluation evidence to define policies rather than assuming a trained detector is ready for autonomous enforcement.

## Defensibility

- A curated corpus of authorized, reproducible long-horizon attack traces.
- Temporal metrics and causal intervention logs.
- Customer-specific regression histories and policy mappings.
- Benchmark credibility through open schemas, transparent negative results, and independent reproductions.
- Integrations and low-noise evidence workflows, not a secret prompt list alone.

## Commercial validation milestones

1. Ten design-partner interviews with AI security/platform teams.
2. Three authorized pilots on sandboxed agent applications.
3. Demonstrate at least one temporal failure missed by single-turn scanning, without cherry-picking.
4. Measure false positives, attack-success reduction, utility preservation, latency, and cost.
5. Convert one pilot to a paid annual evaluation/regression contract before building the full runtime gateway.

## Learned-adversary product rule

A trained HYDRA may eventually prioritize campaign variants that expose a customer model's blind spots. It must remain confined to authorized synthetic campaigns, constrained by the scenario DSL, rate limits, non-destructive tools, and human-approved scope. The product must always retain deterministic scripted suites for repeatability. Learned-HYDRA findings are exploratory until reproduced by a frozen scripted test or independently rerun; adversary novelty is not permission to expand scope.

## Explicit non-goals now

- Claiming enterprise IDS/firewall readiness.
- Scoring real employees or insider risk.
- Autonomous destructive red teaming.
- Broad compliance certification.
- Training a larger model before obtaining better failure telemetry and target integration evidence.
