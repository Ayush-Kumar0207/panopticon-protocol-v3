"""Version-pinned taxonomy helpers for coverage metadata, not certification."""

from __future__ import annotations

from .schemas import TaxonomyReference


OWASP_2025 = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}


def owasp(identifier: str) -> TaxonomyReference:
    if identifier not in OWASP_2025:
        raise KeyError(f"unknown OWASP 2025 identifier: {identifier}")
    return TaxonomyReference(
        framework="OWASP Top 10 for LLM Applications",
        version="2025",
        identifier=identifier,
        name=OWASP_2025[identifier],
        url="https://genai.owasp.org/llm-top-10/",
    )


NIST_AI_600_1 = TaxonomyReference(
    framework="NIST AI RMF Generative AI Profile",
    version="NIST.AI.600-1 (2024)",
    identifier="Information Security",
    name="Information Security risk and risk-management actions",
    url="https://doi.org/10.6028/NIST.AI.600-1",
)
