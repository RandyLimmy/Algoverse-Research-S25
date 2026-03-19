"""Registry of ablation mode presets."""

from __future__ import annotations

from typing import Any, Callable, Dict

from . import hybrid_rag, llm_alone, ontology_rag_cot, rag_cot, scispacy_umls, semantic_rag

PresetBuilder = Callable[..., Dict[str, Any]]

PRESETS: Dict[str, PresetBuilder] = {}

for module in (llm_alone, semantic_rag, hybrid_rag, rag_cot, ontology_rag_cot, scispacy_umls):
    PRESETS.update(module.PRESETS)

__all__ = ["PRESETS", "PresetBuilder"]

