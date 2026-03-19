"""Guided single-diagnosis prompt for LLM-alone baseline."""

from __future__ import annotations

from typing import Sequence


def build_prompt_llm_alone(note: str, hpo_terms: Sequence[str]) -> str:
    return f"""You are a clinical reasoning assistant tasked with identifying rare disease diagnoses based solely on phenotypic features presented in a patient vignette.
Please analyze the following case and provide the most likely diagnosis.

Guardrails:
- Do not invent references or PMIDs. If none available, set PMIDs: N/A.
- Output one specific disease (avoid broad categories unless clearly intended).
- Be specific - avoid generic terms like "intellectual disability", "developmental delay", "congenital abnormality" unless no specific syndrome fits.
- Prefer a canonical/standard disease name if known; include aliases.
- If uncertain among near-synonyms/subtypes, choose the best-supported and most canonical.
- When multiple specific syndromes match, choose the one with strongest phenotypic overlap.

Output format (exact):
Final diagnosis: <DISEASE>
Aliases: <comma-separated synonyms/abbreviations or N/A>
UMLS_CUI: <CUI or N/A>
PMIDs: <comma-separated PMIDs or N/A>
Rationale: <1-2 sentences>

Case:
{note}

HPO terms:
{', '.join(hpo_terms)}
"""

