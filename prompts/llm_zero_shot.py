"""Zero-shot top-5 ranking prompt for LLM-alone runs."""

from __future__ import annotations

from typing import Sequence


def build_prompt_llm_alone_zero_shot_top5(note: str, hpo_terms: Sequence[str]) -> str:
    return f"""You are a clinical reasoning assistant.
Given the patient vignette and HPO terms, list the five most likely rare disease diagnoses, ordered from most to least likely.

Requirements:
- Provide exactly 5 distinct, specific disease names (canonical if possible).
- For each, include known aliases if any; otherwise write N/A.
- Provide a UMLS CUI if known; otherwise N/A.
- Provide supporting PMIDs if known; otherwise N/A.
- Do not invent references.

Format exactly:
1. Final diagnosis: <DISEASE>
   Aliases: <comma-separated or N/A>
   UMLS_CUI: <CUI or N/A>
   PMIDs: <comma-separated or N/A>
   Rationale: <1–2 sentences>
2. ...
3. ...
4. ...
5. ...

Case:
{note}

HPO terms:
{', '.join(hpo_terms)}
"""

