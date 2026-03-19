"""Reconsideration prompt for forced-choice RAG when model picks 'None of the above'."""

from __future__ import annotations

from typing import Sequence


def build_prompt_rag_reconsider(
    note: str,
    hpo_terms: Sequence[str],
    retrieved_docs: Sequence[dict],
    previous_output: str,
    candidates: Sequence[str],
) -> str:
    from rag_scispacy_umls import build_literature_block

    literature = build_literature_block(retrieved_docs or [], include_overlap=True)
    allowed = "\n".join(f"- {c}" for c in (candidates or []))
    prev = previous_output or ""
    return f"""You are a clinical diagnosis assistant. The previous forced-choice decision returned 'None of the above'.

Re-evaluate using the same case, HPO terms, and literature, plus the prior output shown below. If evidence supports a specific disease from the Allowed diagnoses, choose it. Otherwise, if truly unsupported, keep 'None of the above'.

Allowed diagnoses (choose exactly one; output must match text exactly):
{allowed}

Previous output (for reference):
{prev}

Guardrails:
- Cite only PMIDs from Provided Literature; do not invent PMIDs.
- Prefer a canonical/standard disease name; include aliases.
- Avoid generic labels; pick a named syndrome when supported.

Output format (exact):
Final diagnosis: <one item copied verbatim from Allowed diagnoses>
Aliases: <comma-separated synonyms/abbreviations or N/A>
UMLS_CUI: <CUI or N/A>
PMIDs: <comma-separated PMIDs from Provided Literature>
Rationale: <2-4 sentences with PMIDs>

Retrieved Literature (with term overlaps):
{literature}

Case:
{note}

HPO terms:
{', '.join(hpo_terms)}
"""

