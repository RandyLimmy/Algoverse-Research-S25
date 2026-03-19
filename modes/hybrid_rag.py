"""Hybrid RAG (semantic + ontology fusion) ablation configs."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_ROWS = 100
OUTPUT_DIR = "Outputs"
_TOP_K_VALUES = (3, 5, 10)


def _preset(top_k: int):
    def builder(rows: int = DEFAULT_ROWS, output_file: str | None = None, **overrides: Any) -> Dict[str, Any]:
        default_name = output_file or f"{OUTPUT_DIR}/gpt5_hybrid_rag_k{top_k}_n{rows}.csv"
        cfg: Dict[str, Any] = {
            "mode": "hybrid",
            "rows": rows,
            "output_file": default_name,
            "top_k": top_k,
            "pre_k": max(top_k * 6, 60),
            "sem_weight": 0.6,
            "bm25_weight": 0.2,
            "prompt_style": "guided",
            "scaffold": False,
            "force_rag": True,
        }
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cfg

    return builder


PRESETS = {f"hybrid_rag_k{top_k}": _preset(top_k) for top_k in _TOP_K_VALUES}
