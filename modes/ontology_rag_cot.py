"""Ontology-guided chain-of-thought RAG ablation configs."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_ROWS = 100
DEFAULT_TOP_K = 10
OUTPUT_DIR = "Outputs"


def ontology_rag_cot(rows: int = DEFAULT_ROWS, top_k: int = DEFAULT_TOP_K, output_file: str | None = None, **overrides: Any) -> Dict[str, Any]:
    default_name = output_file or f"{OUTPUT_DIR}/gpt5_ontology_rag_cot_n{rows}.csv"
    cfg: Dict[str, Any] = {
        "mode": "hybrid",
        "rows": rows,
        "output_file": default_name,
        "top_k": top_k,
        "pre_k": max(top_k * 6, 60),
        "sem_weight": 0.6,
        "bm25_weight": 0.2,
        "prompt_style": "guided",
        "scaffold": True,
        "force_rag": True,
        "few_shot": True,
        "forced_choice": "auto",
    }
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


PRESETS = {
    "ontology_rag_cot": ontology_rag_cot,
}
