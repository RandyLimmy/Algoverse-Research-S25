"""Ontology-guided RAG + CoT prompt builder."""

from __future__ import annotations

from typing import Sequence


def build_prompt_ontology_cot(note: str, hpo_terms: Sequence[str], retrieved_docs: Sequence[dict]) -> str:
    from rag_scispacy_umls import build_literature_block, extract_gene_like_tokens
    
    literature = build_literature_block(retrieved_docs, include_overlap=True)
    
    # Extract gene hints for additional context
    gene_hints = extract_gene_like_tokens(note)
    gene_section = f"Gene/Cohort Hints: {', '.join(sorted(gene_hints))}" if gene_hints else ""
    
    return f"""You are a clinical reasoning assistant tasked with diagnosing rare diseases using phenotypic features, structured ontology information, and supporting biomedical literature.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT CASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{note}

HPO Terms: {', '.join(hpo_terms)}
{gene_section}

Ontology Scaffold:
Body System(s): Neurologic, Endocrine, etc.
Symptom Categories: Motor, Cognitive, Sensory, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED LITERATURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{literature}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the symptoms, HPO terms, clinical system mappings, and literature to identify the most likely diagnosis. Justify your reasoning using both ontology alignment and the literature (cite key evidence).

Guardrails:
- Cite only PMIDs from the Provided Literature; do not invent PMIDs or evidence.
- Output one specific disease (avoid broad categories unless clearly intended).
- Be specific - avoid generic terms like "intellectual disability", "developmental delay", "congenital abnormality" unless no specific syndrome fits.
- Prefer a canonical/standard disease name if known; include aliases.
- If uncertain among near-synonyms/subtypes, choose the best-supported and most canonical.
- When multiple specific syndromes match, choose the one with strongest phenotypic overlap.

Output format (exact):
Final diagnosis: <DISEASE>
Aliases: <comma-separated synonyms/abbreviations or N/A>
UMLS_CUI: <CUI or N/A>
PMIDs: <comma-separated PMIDs from Provided Literature>
Rationale: <2-4 sentences with PMIDs>
"""

