"""SciSpaCy + UMLS conditional RAG ablation config."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_ROWS = 100
DEFAULT_TOP_K = 10
OUTPUT_DIR = "Outputs"


def scispacy_umls(rows: int = DEFAULT_ROWS, top_k: int = DEFAULT_TOP_K, output_file: str | None = None, **overrides: Any) -> Dict[str, Any]:
    default_name = output_file or f"{OUTPUT_DIR}/gpt5_scispacy_umls_n{rows}.csv"
    cfg: Dict[str, Any] = {
        "mode": "scispacy_umls",
        "rows": rows,
        "output_file": default_name,
        "top_k": top_k,
        "pre_k": max(top_k * 4, 40),
        "prompt_style": "guided",
        "force_rag": True,
    }
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


PRESETS = {
    "scispacy_umls": scispacy_umls,
}
