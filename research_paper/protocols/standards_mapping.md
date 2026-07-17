# Standards and Threat-Taxonomy Mapping

This is a **coverage crosswalk**, not a certification statement. Frameworks evolve; freeze the exact edition/date used in a submitted paper or customer report.

## OWASP Top 10 for LLM Applications (2025)

| OWASP area | Planned Panopticon campaign family | Evidence/metric |
|---|---|---|
| LLM01 Prompt Injection | Direct and indirect instructions that alter agent goals | attack success, unsafe tool call, time to compromise |
| LLM02 Sensitive Information Disclosure | Multi-turn canary/secret exfiltration | exact canary leakage, disclosure severity |
| LLM03 Supply Chain | Poisoned prompt/template/tool description fixture | provenance break, policy deviation |
| LLM04 Data and Model Poisoning | Persistent memory/RAG contamination scenario | delayed compromise, contamination survival |
| LLM05 Improper Output Handling | Model output consumed by mock downstream tools | unsafe sink reached, validation block rate |
| LLM06 Excessive Agency | Over-privileged mock tool and approval-bypass scenario | unauthorized action rate, intervention rate |
| LLM07 System Prompt Leakage | Multi-turn system-instruction extraction | secret match and partial disclosure |
| LLM08 Vector and Embedding Weaknesses | Poisoned retrieval fixture | malicious retrieval influence, citation/source mismatch |
| LLM09 Misinformation | Persistent false premise and decision corruption | incorrect high-impact action, correction latency |
| LLM10 Unbounded Consumption | Recursive/expensive tool plan | token/tool/cost budget breach |

## MITRE ATLAS

Panopticon's future campaign schema can attach ATLAS tactic/technique identifiers. Initial relevant families include LLM prompt injection, AI agent context poisoning, tool invocation, persistence, defense evasion, collection, exfiltration, and impact. Exact identifiers must be resolved against a versioned ATLAS export at release time because ATLAS is a living knowledge base.

## NIST AI RMF Generative AI Profile

The closest risk areas are information security, data privacy, human–AI configuration, information integrity, and value-chain/component integration. Panopticon contributes test evidence; it does not implement the full GOVERN, MAP, MEASURE, and MANAGE organizational program.

## Commercial reporting language

Permitted: “This campaign covers test cases mapped to OWASP LLM01 and LLM06 and records the mapping version.”

Not permitted: “OWASP certified,” “NIST compliant,” or “MITRE approved” unless an actual authorized program grants that status.
