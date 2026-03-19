"""Verification prompt used for the optional second-pass check."""

from __future__ import annotations

from typing import Sequence

from .utils import extract_final_diagnosis, extract_rationale


def build_prompt_verify(
    note: str,
    hpo_terms: Sequence[str],
    retrieved_docs: Sequence[dict] | None,
    prior_output: str,
) -> str:
    literature = ""
    try:
        from rag_scispacy_umls import build_literature_block  # type: ignore

        literature = build_literature_block(retrieved_docs or [], include_overlap=True)
    except Exception:
        literature = "\n\n".join(
            [
                f"PMID {d.get('pmid','')}: {d.get('title','')}\n{d.get('abstract','')}"
                for d in (retrieved_docs or [])
            ]
        )

    proposed_dx = extract_final_diagnosis(prior_output)
    rationale = extract_rationale(prior_output)

    checklist = (
        "Checklist before finalizing:\n"
        "- Does the chosen disease have hallmark features matching the HPO terms?\n"
        "- Are the key features supported by the cited PMIDs in the provided literature?\n"
        "- Are near-miss differentials explicitly less consistent with the case?\n"
        "- Is the disease name canonical, with aliases listed if helpful?\n"
    )

    return (
        f"You are a clinical diagnosis assistant. Verify the previously proposed final diagnosis using the case and retrieved literature.\n\n"
        f"Previously proposed output (for review):\n{prior_output}\n\n"
        f"Proposed diagnosis: {proposed_dx or '<NONE>'}\n"
        f"Proposed rationale: {rationale[:600]}\n\n"
        f"If the proposed diagnosis is not best supported by the literature and symptoms, revise it.\n\n"
        f"Verification {checklist}\n"
        f"Guardrails:\n- Cite only PMIDs from the Provided Literature; do not invent PMIDs or evidence.\n- Output one specific disease (avoid broad categories unless clearly intended).\n- Prefer a canonical/standard disease name if known; include aliases.\n- If uncertain among near-synonyms/subtypes, choose the best-supported and most canonical.\n\n"
        f"Return output in EXACT format:\n"
        f"Final diagnosis: <DISEASE>\n"
        f"Aliases: <comma-separated synonyms/abbreviations or N/A>\n"
        f"UMLS_CUI: <CUI or N/A>\n"
        f"PMIDs: <comma-separated PMIDs from Provided Literature>\n"
        f"Rationale: <2-4 sentences with PMIDs>\n\n"
        f"Retrieved Literature (with overlaps if available):\n{literature}\n\n"
        f"Case:\n{note}\n\n"
        f"HPO terms:\n{', '.join(hpo_terms)}\n"
    )

