"""Main CLI for running GPT-5 diagnosis ablations via preset modes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import llm_diagnosis_pipeline as pipeline
from modes import PRESETS

RUN_PIPELINE_KEYS = {
    "mode",
    "output_file",
    "top_k",
    "pre_k",
    "rows",
    "verify",
    "forced_choice",
    "fc_min_k",
    "fc_max_k",
    "fc_margin",
    "fc_allow_none",
    "scaffold",
    "ce_model",
    "force_rag",
    "sem_weight",
    "bm25_weight",
    "few_shot",
    "prompt_style",
    "use_llm_judge",
    "judge_provider",
    "judge_model",
    "max_llm_calls",
    "verbose_judge",
    "no_overlap_rerank",
    "no_mmr",
    "no_deterministic",
}


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must contain an object at the top level")
    return data


def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in overrides.items():
        if value is None:
            continue
        result[key] = value
    return result


def _build_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.rows is not None:
        overrides["rows"] = args.rows
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.output_file is not None:
        overrides["output_file"] = args.output_file
    if args.prompt_style is not None:
        overrides["prompt_style"] = args.prompt_style
    if args.use_llm_judge:
        overrides["use_llm_judge"] = True
    if args.judge_model is not None:
        overrides["judge_model"] = args.judge_model
    return overrides


def _build_config_from_preset(preset_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    builder = PRESETS[preset_name]
    return builder(**overrides)


def _namespace_from_config(config: Dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "mode": config.get("mode", "llm_alone"),
        "output_file": config.get("output_file", "outputs.csv"),
        "top_k": config.get("top_k", 5),
        "pre_k": config.get("pre_k", 100),
        "rows": config.get("rows", 2),
        "verify": config.get("verify", False),
        "prompt_style": config.get("prompt_style", "guided"),
        "forced_choice": config.get("forced_choice", "off"),
        "fc_min_k": config.get("fc_min_k", 3),
        "fc_max_k": config.get("fc_max_k", 20),
        "fc_margin": config.get("fc_margin", 0.1),
        "fc_allow_none": config.get("fc_allow_none", True),
        "scaffold": config.get("scaffold", False),
        "ce_model": config.get("ce_model"),
        "few_shot": config.get("few_shot", False),
        "force_rag": config.get("force_rag", False),
        "sem_weight": config.get("sem_weight", 0.6),
        "bm25_weight": config.get("bm25_weight", 0.2),
        "no_overlap_rerank": config.get("no_overlap_rerank", False),
        "no_mmr": config.get("no_mmr", False),
        "no_deterministic": config.get("no_deterministic", False),
        "use_llm_judge": config.get("use_llm_judge", False),
        "judge_provider": config.get("judge_provider", "auto"),
        "judge_model": config.get("judge_model"),
        "max_llm_calls": config.get("max_llm_calls", 100),
        "verbose_judge": config.get("verbose_judge", False),
    }
    return argparse.Namespace(**defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-5 diagnosis ablations")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), help="Preset mode to execute")
    parser.add_argument("--config", type=Path, help="Optional JSON config override", default=None)
    parser.add_argument("--rows", type=int, default=None, help="Override number of dataset rows")
    parser.add_argument("--top_k", type=int, default=None, help="Override retrieval top-K")
    parser.add_argument("--output_file", default=None, help="Override output CSV path")
    parser.add_argument("--prompt_style", choices=["guided", "zero_shot_top5"], default=None, help="Override prompt style")
    parser.add_argument("--use_llm_judge", action="store_true", help="Enable LLM judge irrespective of preset")
    parser.add_argument("--judge_model", default=None, help="Override judge model (e.g., gpt-4o, gpt-3.5-turbo)")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config without executing")
    parser.add_argument("--list", action="store_true", help="List available presets and exit")
    args = parser.parse_args()

    if args.list:
        for name in sorted(PRESETS.keys()):
            print(name)
        return

    if not args.config and not args.preset:
        parser.error("Specify --preset or provide a --config file")

    overrides = _build_overrides_from_args(args)

    if args.config:
        config = _load_config(args.config)
        config = _apply_overrides(config, overrides)
    else:
        config = _build_config_from_preset(args.preset, overrides)

    namespace = _namespace_from_config(config)
    pipeline.args = namespace  # type: ignore[attr-defined]

    run_kwargs = {key: config[key] for key in RUN_PIPELINE_KEYS if key in config}

    if args.dry_run:
        print(json.dumps({"config": config, "run_kwargs": run_kwargs}, indent=2))
        return

    # Run the pipeline
    pipeline.run_pipeline(**run_kwargs)
    
    # Automatically compute and save accuracy metrics to CSV
    output_file = config.get("output_file")
    if output_file:
        try:
            print("\n" + "="*60)
            print("Computing accuracy metrics...")
            print("="*60)
            
            from eval_accuracy import compute_accuracy_summary
            
            gold_csv = "Datasets/subset_all_mapped.csv"
            use_llm_judge = config.get("use_llm_judge", False)
            judge_model = config.get("judge_model", "gpt-3.5-turbo")
            judge_provider = config.get("judge_provider", "auto")
            max_llm_calls = config.get("max_llm_calls", 100)
            
            summary = compute_accuracy_summary(
                results_csv=output_file,
                gold_csv=gold_csv,
                use_llm_judge=use_llm_judge,
                judge_model=judge_model,
                judge_provider=judge_provider,
                max_llm_calls=max_llm_calls,
                verbose_judge=False,
            )
            
            # Print summary to console
            total = summary.get("total", 0)
            if total > 0:
                print(f"\n{'='*60}")
                print("ACCURACY SUMMARY")
                print(f"{'='*60}")
                print(f"Total cases: {total}")
                print(f"Strict accuracy: {summary['strict_hits']}/{total} = {summary['strict_pct']*100:.2f}%")
                print(f"Soft accuracy: {summary['soft_hits']}/{total} = {summary['soft_pct']*100:.2f}%")
                print(f"Contains: {summary['contains_hits']}/{total} = {summary['contains_pct']*100:.2f}%")
                if summary.get('top5_soft_hits') is not None:
                    print(f"Top-5 soft: {summary['top5_soft_hits']}/{total} = {summary['top5_soft_pct']*100:.2f}%")
                if use_llm_judge and summary.get('judge_hits', 0) > 0:
                    print(f"LLM Judge: {summary['judge_hits']}/{total} = {summary['judge_pct']*100:.2f}%")
                    print(f"Adjusted soft: {summary['soft_adjusted_hits']}/{total} = {summary['soft_adjusted_pct']*100:.2f}%")
                print(f"{'='*60}")
            
            # Save to accuracy CSV
            import csv
            # Path already imported at module level
            
            # Use dedicated CSV name for improved runs
            accuracy_file = Path("Outputs") / "improved_gpt5_accuracy_metrics.csv"
            
            # Prepare row for CSV
            accuracy_row = {
                "file": Path(output_file).name,
                "total": total,
                "strict": f"{summary['strict_hits']}/{total} = {summary['strict_pct']*100:.2f}%",
                "soft": f"{summary['soft_hits']}/{total} = {summary['soft_pct']*100:.2f}%",
                "contains": f"{summary['contains_hits']}/{total} = {summary['contains_pct']*100:.2f}%",
                "coverage": f"{summary['coverage_hits']}/{total} = {summary['coverage_pct']*100:.2f}%",
                "top5_strict": f"{summary.get('top5_strict_hits', 0)}/{total} = {(summary.get('top5_strict_pct') or 0)*100:.2f}%" if summary.get('top5_strict_hits') is not None else "",
                "top5_soft": f"{summary.get('top5_soft_hits', 0)}/{total} = {(summary.get('top5_soft_pct') or 0)*100:.2f}%" if summary.get('top5_soft_hits') is not None else "",
                "llm_judge": f"{summary.get('judge_hits', 0)}/{total} = {summary.get('judge_pct', 0)*100:.2f}%" if use_llm_judge else "",
                "strict_adjusted": f"{summary['strict_adjusted_hits']}/{total} = {summary['strict_adjusted_pct']*100:.2f}%",
                "soft_adjusted": f"{summary['soft_adjusted_hits']}/{total} = {summary['soft_adjusted_pct']*100:.2f}%",
            }
            
            # Check if file exists to determine if we need header
            file_exists = accuracy_file.exists()
            
            # Append to CSV
            with accuracy_file.open("a", newline="", encoding="utf-8") as f:
                fieldnames = ["file", "total", "strict", "soft", "contains", "coverage", 
                             "top5_strict", "top5_soft", "llm_judge", "strict_adjusted", "soft_adjusted"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(accuracy_row)
            
            print(f"\n✅ Accuracy metrics saved to: {accuracy_file}")
            
        except Exception as e:
            print(f"\n⚠️  Failed to compute accuracy metrics: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
