"""Configuration presets for LLM-alone ablations."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_ROWS = 100
OUTPUT_DIR = "Outputs"


def _merge(base: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = base.copy()
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def zero_shot_top5(rows: int = DEFAULT_ROWS, output_file: str | None = None, **overrides: Any) -> Dict[str, Any]:
    default_name = output_file or f"{OUTPUT_DIR}/gpt5_llm_zero_shot_top5_n{rows}.csv"
    base = {
        "mode": "llm_alone",
        "rows": rows,
        "output_file": default_name,
        "prompt_style": "zero_shot_top5",
        "scaffold": False,
    }
    return _merge(base, overrides)


def guided(rows: int = DEFAULT_ROWS, output_file: str | None = None, **overrides: Any) -> Dict[str, Any]:
    default_name = output_file or f"{OUTPUT_DIR}/gpt5_llm_guided_n{rows}.csv"
    base = {
        "mode": "llm_alone",
        "rows": rows,
        "output_file": default_name,
        "prompt_style": "guided",
        "scaffold": False,
    }
    return _merge(base, overrides)


def guided_ontology(rows: int = DEFAULT_ROWS, output_file: str | None = None, **overrides: Any) -> Dict[str, Any]:
    default_name = output_file or f"{OUTPUT_DIR}/gpt5_llm_guided_ontology_n{rows}.csv"
    base = {
        "mode": "llm_alone",
        "rows": rows,
        "output_file": default_name,
        "prompt_style": "guided",
        "scaffold": True,
    }
    return _merge(base, overrides)


PRESETS = {
    "llm_zero_shot_top5": zero_shot_top5,
    "llm_guided": guided,
    "llm_guided_ontology": guided_ontology,
}
