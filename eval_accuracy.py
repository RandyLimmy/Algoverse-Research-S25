import argparse
import os
import re
import json
from typing import Optional, Tuple, Dict, Any

import pandas as pd
try:
    from rag_scispacy_umls import canonicalize_disease_name  # type: ignore
except Exception:
    canonicalize_disease_name = None


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[`“”\"']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def extract_final_dx(output_text: str) -> str:
    if not isinstance(output_text, str):
        return ""
    m = re.search(r"final\s+diagnosis\s*[:\-–]?\s*\**\s*([^\n]+)", output_text, flags=re.I)
    return (m.group(1).strip() if m else "")


def _maybe_load_openai():
    """Load OpenAI client if available and API key is set."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    try:
        import openai  # type: ignore
    except Exception:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def _maybe_load_gemini(model: str):
    """Load Gemini model if available and API key is set.

    Returns a tuple ("gemini", model_instance) or None on failure.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        return ("gemini", model_obj)
    except Exception:
        return None


def _extract_rationale_text(output_text: str, max_chars: int = 700) -> str:
    """Extract a rationale snippet from an LLM output, if present."""
    if not isinstance(output_text, str) or not output_text.strip():
        return ""
    try:
        m = re.search(r"rationale\s*[:\-–]?\s*([\s\S]+)$", output_text, flags=re.I)
        text = m.group(1).strip() if m else output_text
        return text[:max_chars]
    except Exception:
        return output_text[:max_chars]


def _judge_equivalence_llm(client: Tuple[str, Any] | Any, model: str, name_a: str, name_b: str, rationale_a: str = "") -> Optional[bool]:
    """Ask the LLM to judge if two disease names refer to the same specific disease.

    Returns True/False, or None on error.
    """
    if not client:
        return None
    try:
        context = (f"\nRationale for Name A (predicted):\n{rationale_a}\n" if rationale_a else "")
        prompt = (
            "You are a medical terminology expert. Decide if the two names refer to the same specific disease.\n"
            "Rules:\n"
            "- Answer ONLY YES or NO.\n"
            "- Treat synonyms, acronyms, eponyms, spelling variants, and hyphenation variants as the SAME disease.\n"
            "- If one is a broad class and the other a subtype (or vice versa), answer NO.\n"
            "- If they are distinct subtypes, answer NO.\n\n"
            f"Name A (predicted): {name_a}\n"
            f"Name B (gold): {name_b}\n"
            f"{context}"
            "Answer:"
        )
        provider = None
        client_obj = client
        if isinstance(client, tuple) and len(client) == 2:
            provider, client_obj = client
        if provider == "openai" or (provider is None and hasattr(client_obj, "chat")):
            resp = client_obj.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = (resp.choices[0].message.content or "").strip().upper()
        elif provider == "gemini" or (provider is None and hasattr(client_obj, "generate_content")):
            resp = client_obj.generate_content(prompt)
            try:
                txt = (resp.text or "").strip().upper()
            except Exception:
                txt = (str(resp) or "").strip().upper()
        else:
            return None
        if "YES" in txt.split():
            return True
        if "NO" in txt.split():
            return False
        return True if txt.startswith("YES") else (False if txt.startswith("NO") else None)
    except Exception:
        return None


def _evaluate_accuracy(
    results_csv: str,
    gold_csv: str,
    nrows: int | None = None,
    use_llm_judge: bool = False,
    judge_model: str = "gpt-3.5-turbo",
    max_llm_calls: int = 100,
    verbose_judge: bool = False,
    judge_provider: str = "auto",
    return_per_strategy: bool = False,
):
    df = pd.read_csv(results_csv)
    # Align by row_id if present to handle random sampling
    if "row_id" in df.columns:
        g_full = pd.read_csv(gold_csv)
        try:
            g = g_full.iloc[df["row_id"].astype(int).tolist()].reset_index(drop=True)
        except Exception:
            g = g_full.head(len(df))
    else:
        if nrows is not None:
            df = df.head(nrows)
        g = pd.read_csv(gold_csv).head(len(df))

    # Prefer explicit final_diagnosis column if present
    if "final_diagnosis" in df.columns:
        raw_preds = [str(x) for x in df.get("final_diagnosis", [])]
    else:
        raw_preds = [extract_final_dx(x) for x in df.get("llm_output", [])]
    # Canonicalize both sides if helper is available
    if canonicalize_disease_name is not None:
        raw_preds = [canonicalize_disease_name(x) for x in raw_preds]
        raw_golds = [canonicalize_disease_name(str(x)) for x in g.get("disease", [])]
    else:
        raw_golds = [str(x) for x in g.get("disease", [])]
    preds = [normalize_text(x) for x in raw_preds]
    golds = [normalize_text(str(x)) for x in raw_golds]

    total = len(preds)
    strict_hits = sum(1 for p, l in zip(preds, golds) if p and p == l)

    def toks(s: str) -> set[str]:
        return set([t for t in s.split() if t])

    def jacc(a: set[str], b: set[str]) -> float:
        return (len(a & b) / len(a | b)) if (a or b) else 0.0

    soft_hits = 0
    contains_hits = 0
    coverage_hits = 0
    judge_nonsoft_hits = 0
    # Optional LLM judge
    client_tuple: Tuple[str, Any] | Any | None = None
    model_to_use = judge_model
    if use_llm_judge:
        if judge_provider == "openai":
            oc = _maybe_load_openai()
            client_tuple = ("openai", oc) if oc is not None else None
            if (not judge_model) or judge_model.lower().startswith("gemini"):
                model_to_use = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        elif judge_provider == "gemini":
            if (not judge_model) or judge_model.lower().startswith("gpt-"):
                model_to_use = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            client_tuple = _maybe_load_gemini(model_to_use)
        else:
            oc = _maybe_load_openai()
            if oc is not None:
                client_tuple = ("openai", oc)
                if (not judge_model) or judge_model.lower().startswith("gemini"):
                    model_to_use = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            else:
                model_to_use = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") if ((not judge_model) or judge_model.lower().startswith("gpt-")) else judge_model
                client_tuple = _maybe_load_gemini(model_to_use)
        if client_tuple is None:
            print("LLM judge requested but no client available (check OPENAI_API_KEY or GEMINI_API_KEY). Skipping judge.")
    judge_hits = 0
    llm_calls = 0
    cache: dict[Tuple[str, str], bool] = {}

    outputs_col = list(df.get("llm_output", []))
    outputs_txt = [str(x) if pd.notna(x) else "" for x in outputs_col]
    strategies = [str(x) if pd.notna(x) else "" for x in df.get("strategy", [])]
    per_strategy: Dict[str, Dict[str, int]] = {}
    def bump(strategy: str, key: str):
        per_strategy.setdefault(strategy or "", {}).setdefault(key, 0)
        per_strategy[strategy or ""][key] += 1

    for idx, (raw_p, raw_l, p, l, out_txt, strat) in enumerate(zip(raw_preds, raw_golds, preds, golds, outputs_txt, strategies)):
        if not p or not l:
            continue
        # Soft match flag for this row
        is_soft = False
        if p in l or l in p or jacc(toks(p), toks(l)) >= 0.5:
            is_soft = True
            soft_hits += 1
            bump(strat, "soft")
        p_norm = normalize_text("" if p is None else p)
        if l and l in p_norm:
            contains_hits += 1
            bump(strat, "contains")
        # coverage (did candidate include gold?) — here proxy by contains
        if l in p_norm:
            coverage_hits += 1
            bump(strat, "coverage")
        # If strict already matched, no need for judge
        if p == l:
            bump(strat, "strict")
            continue
        # LLM judge on unresolved cases
        if use_llm_judge and client_tuple is not None and llm_calls < max_llm_calls:
            key = (raw_p.strip().lower(), raw_l.strip().lower())
            if key in cache:
                same = cache[key]
            else:
                rationale_snippet = _extract_rationale_text(out_txt)
                same = _judge_equivalence_llm(client_tuple, model_to_use, raw_p, raw_l, rationale_snippet) or False
                cache[key] = same
                llm_calls += 1
            if verbose_judge:
                print(f"[JUDGE] row={idx+1} pred='{raw_p}' gold='{raw_l}' -> {'YES' if same else 'NO'}")
            if same:
                judge_hits += 1
                bump(strat, "judge")
                # Count only if this row was NOT already a soft match
                if not is_soft:
                    judge_nonsoft_hits += 1

    topk_hits = 0
    topk_soft_hits = 0
    if "ranked_diagnoses" in df.columns:
        for ranked_raw, gold_raw in zip(df.get("ranked_diagnoses", []), raw_golds):
            if not pd.notna(ranked_raw):
                continue
            try:
                ranked_list = json.loads(ranked_raw) if isinstance(ranked_raw, str) else list(ranked_raw)
            except Exception:
                continue
            if not ranked_list:
                continue
            gold_norm = normalize_text(str(gold_raw))
            gold_tokens = toks(gold_norm)
            for cand in ranked_list:
                text = str(cand)
                if canonicalize_disease_name is not None:
                    text = canonicalize_disease_name(text)
                cand_norm = normalize_text(text)
                if not cand_norm:
                    continue
                if gold_norm and cand_norm == gold_norm:
                    topk_hits += 1
                    topk_soft_hits += 1
                    break
                cand_tokens = toks(cand_norm)
                if gold_norm and jacc(cand_tokens, gold_tokens) >= 0.5:
                    topk_soft_hits += 1
                    break

    provider_name = None
    if isinstance(client_tuple, tuple):
        provider_name = client_tuple[0]

    summary: Dict[str, Any] = {
        "total": total,
        "strict_hits": strict_hits,
        "strict_pct": (strict_hits / total) if total else 0.0,
        "soft_hits": soft_hits,
        "soft_pct": (soft_hits / total) if total else 0.0,
        "contains_hits": contains_hits,
        "contains_pct": (contains_hits / total) if total else 0.0,
        "coverage_hits": coverage_hits,
        "coverage_pct": (coverage_hits / total) if total else 0.0,
        "top5_strict_hits": topk_hits if "ranked_diagnoses" in df.columns else None,
        "top5_strict_pct": ((topk_hits / total) if total else 0.0) if "ranked_diagnoses" in df.columns else None,
        "top5_soft_hits": topk_soft_hits if "ranked_diagnoses" in df.columns else None,
        "top5_soft_pct": ((topk_soft_hits / total) if total else 0.0) if "ranked_diagnoses" in df.columns else None,
        "judge_hits": judge_hits if use_llm_judge else 0,
        "judge_pct": ((judge_hits / total) if total else 0.0) if use_llm_judge else 0.0,
        "strict_adjusted_hits": strict_hits + (judge_hits if use_llm_judge else 0),
        "strict_adjusted_pct": ((strict_hits + (judge_hits if use_llm_judge else 0)) / total) if total else 0.0,
        "soft_adjusted_hits": soft_hits + (judge_nonsoft_hits if use_llm_judge else 0),
        "soft_adjusted_pct": ((soft_hits + (judge_nonsoft_hits if use_llm_judge else 0)) / total) if total else 0.0,
        "llm_judge_calls": llm_calls if use_llm_judge else 0,
        "judge_provider": provider_name or "",
        "judge_model": model_to_use if use_llm_judge else "",
    }

    if total == 0:
        if return_per_strategy:
            return summary, per_strategy
        return summary, None

    print(f"STRICT {strict_hits}/{total} = {strict_hits/total*100:.2f}%")
    print(f"SOFT   {soft_hits}/{total} = {soft_hits/total*100:.2f}%")
    print(f"CONTAINS {contains_hits}/{total} = {contains_hits/total*100:.2f}%")
    print(f"COVERAGE (proxy) {coverage_hits}/{total} = {coverage_hits/total*100:.2f}%")
    if "ranked_diagnoses" in df.columns:
        print(f"TOP5_STRICT {topk_hits}/{total} = {topk_hits/total*100:.2f}%")
        print(f"TOP5_SOFT {topk_soft_hits}/{total} = {topk_soft_hits/total*100:.2f}%")

    if return_per_strategy:
        return summary, per_strategy
    return summary, None


def compute_accuracy(
    results_csv: str,
    gold_csv: str,
    nrows: int | None = None,
    use_llm_judge: bool = False,
    judge_model: str = "gpt-3.5-turbo",
    max_llm_calls: int = 100,
    verbose_judge: bool = False,
    judge_provider: str = "auto",
):
    summary, per_strategy = _evaluate_accuracy(
        results_csv,
        gold_csv,
        nrows,
        use_llm_judge,
        judge_model,
        max_llm_calls,
        verbose_judge,
        judge_provider,
        return_per_strategy=True,
    )

    total = summary.get("total", 0) or 1  # avoid divide by zero in prints

    print(
        f"STRICT {summary['strict_hits']}/{summary['total']} = {summary['strict_pct']*100:.2f}%"
    )
    print(
        f"SOFT   {summary['soft_hits']}/{summary['total']} = {summary['soft_pct']*100:.2f}%"
    )
    print(
        f"CONTAINS {summary['contains_hits']}/{summary['total']} = {summary['contains_pct']*100:.2f}%"
    )
    print(
        f"COVERAGE (proxy) {summary['coverage_hits']}/{summary['total']} = {summary['coverage_pct']*100:.2f}%"
    )
    if summary["top5_strict_hits"] is not None:
        print(
            f"TOP5_STRICT {summary['top5_strict_hits']}/{summary['total']} = {summary['top5_strict_pct']*100:.2f}%"
        )
        print(
            f"TOP5_SOFT {summary['top5_soft_hits']}/{summary['total']} = {summary['top5_soft_pct']*100:.2f}%"
        )
    if use_llm_judge:
        print(
            f"LLM_JUDGE {summary['judge_hits']}/{summary['total']} = {summary['judge_pct']*100:.2f}% (calls={summary['llm_judge_calls']}, provider={summary['judge_provider'] or 'n/a'}, model={summary['judge_model'] or 'n/a'})"
        )
        print(
            f"ADJUSTED (STRICT+LLM_JUDGE) {summary['strict_adjusted_hits']}/{summary['total']} = {summary['strict_adjusted_pct']*100:.2f}%"
        )
        print(
            f"SOFT_ADJUSTED (SOFT+LLM_JUDGE on non-soft) {summary['soft_adjusted_hits']}/{summary['total']} = {summary['soft_adjusted_pct']*100:.2f}%"
        )

    if per_strategy:
        print("\nPer-strategy counts (not normalized):")
        for strat, counts in per_strategy.items():
            print(f"- {strat or 'unspecified'}: {counts}")

    return summary


def compute_accuracy_summary(
    results_csv: str,
    gold_csv: str = "Datasets/subset_all_mapped.csv",
    nrows: int | None = None,
    use_llm_judge: bool = False,
    judge_model: str = "gpt-3.5-turbo",
    max_llm_calls: int = 100,
    verbose_judge: bool = False,
    judge_provider: str = "auto",
) -> Dict[str, Any]:
    summary, _ = _evaluate_accuracy(
        results_csv,
        gold_csv,
        nrows,
        use_llm_judge,
        judge_model,
        max_llm_calls,
        verbose_judge,
        judge_provider,
        return_per_strategy=False,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results CSV with llm_output column")
    ap.add_argument("--gold", default="Datasets/subset_all_mapped.csv", help="Path to gold CSV with disease column")
    ap.add_argument("--nrows", type=int, default=None, help="Limit evaluation to first N rows")
    ap.add_argument("--use_llm_judge", action="store_true", help="Enable LLM-based equivalence judge for unresolved cases")
    ap.add_argument("--judge_model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), help="Model for LLM judge (OpenAI or Gemini)")
    ap.add_argument("--max_llm_calls", type=int, default=100, help="Max number of LLM judge calls")
    ap.add_argument("--verbose_judge", action="store_true", help="Print per-case LLM judge decisions to terminal")
    ap.add_argument("--judge_provider", choices=["auto", "openai", "gemini"], default="auto", help="Which provider to use for the LLM judge")
    args = ap.parse_args()
    compute_accuracy(
        args.results,
        args.gold,
        args.nrows,
        args.use_llm_judge,
        args.judge_model,
        args.max_llm_calls,
        args.verbose_judge,
        args.judge_provider,
    )


if __name__ == "__main__":
    main()
