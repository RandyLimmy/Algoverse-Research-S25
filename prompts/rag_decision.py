"""Decision-layer RAG prompt that enforces single-diagnosis output."""

from __future__ import annotations

from typing import Sequence


def build_prompt_rag_decision(note: str, hpo_terms: Sequence[str], retrieved_docs: Sequence[dict]) -> str:
    from rag_scispacy_umls import build_literature_block, extract_gene_like_tokens

    literature = build_literature_block(retrieved_docs, include_overlap=True)
    
    # Extract gene hints for additional context
    gene_hints = extract_gene_like_tokens(note)
    gene_section = f"Gene/Cohort Hints: {', '.join(sorted(gene_hints))}" if gene_hints else ""
    
    return f"""You are a clinical diagnosis assistant. Use the case details and the retrieved literature to select a SINGLE most likely diagnosis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT CASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{note}

HPO Terms: {', '.join(hpo_terms)}
{gene_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED LITERATURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{literature}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Guardrails:
- Cite only PMIDs from the Provided Literature; do not invent PMIDs or evidence.
- Output one specific disease (avoid broad categories unless clearly intended).
- Be specific: avoid generic labels like "intellectual disability", "developmental delay", or "congenital abnormality"; pick a named syndrome.
- Prefer a canonical/standard disease name if known; include aliases.
- If uncertain among near-synonyms/subtypes, choose the best-supported and most canonical.
- Ensure the cited PMIDs explicitly support the chosen disease.

Output format (exact):
Final diagnosis: <DISEASE>
Aliases: <comma-separated synonyms/abbreviations or N/A>
UMLS_CUI: <CUI or N/A>
PMIDs: <comma-separated PMIDs from Provided Literature>
Rationale: <2-4 sentences with PMIDs>
"""

