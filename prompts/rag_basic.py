"""Standard RAG reasoning prompt without ontology scaffold."""

from __future__ import annotations

from typing import Sequence


def build_prompt_rag(note: str, hpo_terms: Sequence[str], retrieved_docs: Sequence[dict]) -> str:
    # Build literature block using updated function from rag_scispacy_umls
    from rag_scispacy_umls import build_literature_block, extract_gene_like_tokens
    
    literature = build_literature_block(retrieved_docs, include_overlap=True)
    
    # Extract gene hints for additional context
    gene_hints = extract_gene_like_tokens(note)
    gene_section = f"Gene/Cohort Hints: {', '.join(sorted(gene_hints))}" if gene_hints else ""
    
    return f"""You are a clinical reasoning assistant for rare disease diagnosis.

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
Use both the case details and the literature to identify the most likely diagnosis. Reference specific literature passages when appropriate.

Guardrails:
- Cite only PMIDs from the Provided Literature; do not invent PMIDs or evidence.
- Output one specific disease (avoid broad categories unless clearly intended).
- Prefer a canonical/standard disease name if known; include aliases.
- If uncertain among near-synonyms/subtypes, choose the best-supported and most canonical.

Output format (exact):
Final diagnosis: <DISEASE>
Aliases: <comma-separated synonyms/abbreviations or N/A>
UMLS_CUI: <CUI or N/A>
PMIDs: <comma-separated PMIDs from Provided Literature>
Rationale: <2-4 sentences with PMIDs>
"""

