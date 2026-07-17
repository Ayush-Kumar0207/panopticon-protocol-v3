# Ethics, Dual Use, and Responsible Release

## Purpose

Panopticon is a synthetic environment for studying agent oversight, hidden state, deceptive evidence, and hard safety constraints. Its fictional counter-espionage framing is a narrative device, not a validated model of real organizations or people.

## Potential benefits

- More rigorous evaluation of long-horizon agent failures.
- Explicit separation of utility and non-negotiable constraints.
- Auditable provenance for repair, fallback, and supervision.
- Defensive red-team regression testing before agent deployment.
- Better evidence for when automated agents should be restricted or escalated to humans.

## Risks

1. **Surveillance framing:** Users could misapply worker suspicion concepts to real employees.
2. **Dual use:** Red-team scenarios can reveal attack patterns as well as defenses.
3. **Automation bias:** A numerical risk score may be treated as truth.
4. **False confidence:** A passed simulation may be marketed as general safety.
5. **Sensitive logging:** Real target adapters could collect prompts, secrets, documents, or tool outputs.
6. **Excessive agency:** A test runner with broad tools could itself cause harmful side effects.
7. **Taxonomy laundering:** Mapping to OWASP, MITRE, or NIST does not constitute certification or compliance.

## Required controls for external-target testing

- Written authorization and explicit target scope.
- Separate test tenant and synthetic canary data.
- Default-deny tool permissions and spend/rate limits.
- No destructive actions; use mock/sandbox tools where possible.
- Secret redaction, encrypted storage, retention limits, and access audit.
- Kill switch and per-campaign budget.
- Human approval for high-impact tool actions.
- Incident process for unexpected real data exposure.
- Clear deletion/export controls for customer evidence.

## Human impact

Never use Panopticon to infer employee loyalty, criminal intent, insider status, or suitability. Any future human-subject experiment requires informed consent, institutional/ethics review where applicable, data minimization, and an appeal/remediation process.

## Responsible claim policy

Use “passed the specified sampled simulation gate,” not “safe.” Use “taxonomy-aligned test coverage,” not “OWASP/NIST compliant.” Publish negative results and intervention dependence. Document where evidence is synthetic, preliminary, or planned.

## Release tiers

- **Open:** environment, schemas, defensive metrics, mock scenarios, aggregate results.
- **Controlled:** realistic customer traces, sensitive prompts, exact exploit chains, credentials, or target-specific evidence.
- **Do not release:** live secrets, personal data, destructive payloads, or unauthorized target results.
