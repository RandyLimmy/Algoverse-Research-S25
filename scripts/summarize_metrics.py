"""Aggregate strict/soft/top-k/LLM judge metrics across output CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable

import os
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_accuracy import compute_accuracy_summary


DEFAULT_FIELDS = [
    "file",
    "total",
    "strict",
    "soft",
    "contains",
    "coverage",
    "top5_strict",
    "top5_soft",
    "llm_judge",
    "strict_adjusted",
    "soft_adjusted",
]


def _iter_csv_files(directory: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() == ".csv":
            yield path


def _format_metric(hits: int | None, total: int) -> str:
    if hits is None or total == 0:
        return ""
    return f"{hits}/{total} = {hits/total*100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize accuracy metrics for result CSVs")
    parser.add_argument("--results-dir", default="Outputs", help="Directory containing results CSVs")
    parser.add_argument("--pattern", default="gpt5_*.csv", help="Glob pattern for result CSVs")
    parser.add_argument("--gold", default="Datasets/subset_all_mapped.csv", help="Gold CSV with disease labels")
    parser.add_argument("--output", default="Outputs/metrics_summary.csv", help="Path to write summary CSV")
    parser.add_argument("--use_llm_judge", action="store_true", help="Include LLM judge when computing metrics")
    parser.add_argument("--judge_model", default=None, help="Model name for LLM judge (defaults to env)" )
    parser.add_argument("--max_llm_calls", type=int, default=100, help="Max LLM judge calls per file")
    parser.add_argument("--judge_provider", choices=["auto", "openai", "gemini"], default="auto", help="Provider for LLM judge")
    parser.add_argument("--verbose", action="store_true", help="Print per-file summaries to stdout")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    files = list(_iter_csv_files(results_dir, args.pattern))
    if not files:
        raise SystemExit(f"No CSV files matching '{args.pattern}' found in {results_dir}")

    summaries: list[Dict[str, Any]] = []
    for csv_path in files:
        summary = compute_accuracy_summary(
            results_csv=str(csv_path),
            gold_csv=args.gold,
            use_llm_judge=args.use_llm_judge,
            judge_model=args.judge_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_llm_calls=args.max_llm_calls,
            verbose_judge=False,
            judge_provider=args.judge_provider,
        )
        total = summary.get("total", 0)
        summary_row: Dict[str, Any] = {
            "file": csv_path.name,
            "total": total,
            "strict": _format_metric(summary.get("strict_hits"), total),
            "soft": _format_metric(summary.get("soft_hits"), total),
            "contains": _format_metric(summary.get("contains_hits"), total),
            "coverage": _format_metric(summary.get("coverage_hits"), total),
            "top5_strict": _format_metric(summary.get("top5_strict_hits"), total) if summary.get("top5_strict_hits") is not None else "",
            "top5_soft": _format_metric(summary.get("top5_soft_hits"), total) if summary.get("top5_soft_hits") is not None else "",
            "llm_judge": _format_metric(summary.get("judge_hits"), total) if args.use_llm_judge else "",
            "strict_adjusted": _format_metric(summary.get("strict_adjusted_hits"), total),
            "soft_adjusted": _format_metric(summary.get("soft_adjusted_hits"), total),
        }
        summaries.append(summary_row)
        if args.verbose:
            strict_str = summary_row["strict"] or "n/a"
            soft_str = summary_row["soft"] or "n/a"
            print(f"Processed {csv_path.name}: strict={strict_str} soft={soft_str}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = DEFAULT_FIELDS
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    if args.verbose:
        print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
