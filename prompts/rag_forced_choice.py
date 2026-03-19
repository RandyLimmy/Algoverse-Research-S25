"""Forced-choice RAG prompt when candidate list is pre-defined."""

from __future__ import annotations

from typing import Sequence


def build_prompt_rag_forced_choice(
    note: str,
    hpo_terms: Sequence[str],
    retrieved_docs: Sequence[dict],
    candidates: Sequence[str],
) -> str:
    from rag_scispacy_umls import build_literature_block, extract_gene_like_tokens

    literature = build_literature_block(retrieved_docs, include_overlap=True)
    allowed = "\n".join(f"- {c}" for c in candidates)
    
    # Extract gene hints for additional context
    gene_hints = extract_gene_like_tokens(note)
    gene_section = f"Gene/Cohort Hints: {', '.join(sorted(gene_hints))}" if gene_hints else ""
    
    return f"""You are a clinical diagnosis assistant. Choose ONE final diagnosis strictly from the Allowed diagnoses list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT CASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{note}

HPO Terms: {', '.join(hpo_terms)}
{gene_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALLOWED DIAGNOSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Choose exactly one from this list (output must match text exactly):
{allowed}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED LITERATURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{literature}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Steps:
1) List key symptoms and mapped HPO terms.
2) From literature, compare candidate diseases and cite PMIDs.
3) Pick ONE final diagnosis from Allowed diagnoses and justify briefly.

Guardrails:
- Choose exactly one from the Allowed diagnoses; do not output anything else.
- Cite only PMIDs from the Provided Literature; do not invent PMIDs or evidence.
- Prefer a canonical/standard disease name if variants appear in Allowed list.

Output format (exact):
Final diagnosis: <one item copied verbatim from Allowed diagnoses>
Aliases: <comma-separated synonyms/abbreviations or N/A>
UMLS_CUI: <CUI or N/A>
PMIDs: <comma-separated PMIDs from Provided Literature>
Rationale: <2-4 sentences with PMIDs>
"""

