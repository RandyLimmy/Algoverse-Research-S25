"""Utility to list or execute the full GPT-5 ablation preset suite."""

from __future__ import annotations

import argparse
import subprocess

PRESET_ALIASES = [
    ("llm_zero_shot_top5", "llm_zero_shot_top5"),
    ("llm_guided", "llm_guided"),
    ("llm_guided_ontology", "llm_guided_ontology"),
    ("semantic_rag_k3", "semantic_rag_k3"),
    ("semantic_rag_k5", "semantic_rag_k5"),
    ("semantic_rag_k10", "semantic_rag_k10"),
    ("hybrid_rag_k3", "hybrid_rag_k3"),
    ("hybrid_rag_k5", "hybrid_rag_k5"),
    ("hybrid_rag_k10", "hybrid_rag_k10"),
    ("rag_no_cot", "rag_cot"),
    ("ontology_rag_no_cot", "ontology_rag_cot"),
    ("scispacy_umls", "scispacy_umls"),
]

ALIAS_TO_PRESET = {alias: actual for alias, actual in PRESET_ALIASES}
PRESETS = [alias for alias, _ in PRESET_ALIASES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or list GPT-5 ablation presets")
    parser.add_argument("--execute", action="store_true", help="Execute each preset sequentially")
    parser.add_argument("--start", default=None, help="Preset name to resume from")
    parser.add_argument("--rows", type=int, default=None, help="Override rows for all presets")
    parser.add_argument("--use_llm_judge", action="store_true", help="Enable LLM judge for all presets")
    parser.add_argument(
        "--suffix",
        default="_no_cot",
        help="Optional suffix appended to output filenames (set to '' to keep default preset names)",
    )
    args = parser.parse_args()

    presets = PRESETS
    if args.start:
        actual_start = ALIAS_TO_PRESET.get(args.start, args.start)
        if args.start not in presets:
            raise SystemExit(f"Unknown start preset: {args.start}")
        start_index = presets.index(args.start)
        presets = presets[start_index:]

    for preset in presets:
        actual_preset = ALIAS_TO_PRESET.get(preset, preset)
        cmd = ["python3", "run_ablations.py", "--preset", actual_preset]
        if args.rows is not None:
            cmd.extend(["--rows", str(args.rows)])
        if args.use_llm_judge:
            cmd.append("--use_llm_judge")
        if args.suffix is not None:
            suffix = args.suffix
            if suffix and not suffix.startswith("_"):
                suffix = f"_{suffix}"
            assumed_rows = args.rows if args.rows is not None else 100
            base_name = {
                "rag_cot": "rag",
                "ontology_rag_cot": "ontology_rag",
            }.get(actual_preset, preset)
            output_file = (
                f"Outputs/gpt5_{base_name}{suffix}_n{assumed_rows}.csv"
                if suffix
                else f"Outputs/gpt5_{base_name}_n{assumed_rows}.csv"
            )
            cmd.extend(["--output_file", output_file])
        if args.execute:
            subprocess.run(cmd, check=True)
        else:
            print(" ".join(cmd))


if __name__ == "__main__":
    main()
