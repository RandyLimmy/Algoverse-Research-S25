# ⚠️ Make sure to use transformers==4.39.3 and torch==2.2.2
import os
import time
import re
import warnings
import pandas as pd
import ast
import json
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
from dotenv import load_dotenv
import argparse
try:
    # Optional: used for canonicalizing disease names in metrics
    from rag_scispacy_umls import canonicalize_disease_name  # type: ignore
except Exception:
    canonicalize_disease_name = None
# Optional: end-of-run evaluator with LLM judge
try:
    from eval_accuracy import compute_accuracy as _eval_compute_accuracy  # type: ignore
except Exception:
    _eval_compute_accuracy = None
# from tqdm import tqdm

from prompts import (
    build_prompt_llm_alone,
    build_prompt_llm_alone_zero_shot_top5,
    build_prompt_ontology_cot,
    build_prompt_rag,
    build_prompt_rag_decision,
    build_prompt_rag_forced_choice,
    build_prompt_rag_reconsider,
    build_prompt_verify,
)
from prompts.utils import (
    extract_final_diagnosis,
    extract_rationale,
    extract_ranked_diagnoses,
    normalize_text,
    serialize_ranked_diagnoses,
)

# Load environment variables from .env if present
load_dotenv()

# Suppress noisy warnings (sklearn model persistence, HF/regex future warnings)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# 💬 LLM CALLER (OpenAI or Gemini)
# -----------------------
_GEMINI_READY = False
_OPENAI_READY = False
_OPENAI_CLIENT = None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # auto|openai|gemini

def _ensure_gemini():
    global _GEMINI_READY
    if _GEMINI_READY:
        return
    if genai is None:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add GEMINI_API_KEY=... to your .env or shell.")
    genai.configure(api_key=key)
    _GEMINI_READY = True


def _ensure_openai():
    global _OPENAI_READY, _OPENAI_CLIENT
    if _OPENAI_READY:
        return
    if OpenAI is None:
        raise RuntimeError("openai not installed. Run: pip install openai")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add OPENAI_API_KEY=... to your .env or shell.")
    _OPENAI_CLIENT = OpenAI(api_key=key)
    _OPENAI_READY = True


def _current_llm_config() -> tuple[str, str]:
    provider = _LLM_PROVIDER
    if provider == "auto":
        provider = "openai" if os.getenv("OPENAI_API_KEY") else ("gemini" if os.getenv("GEMINI_API_KEY") else "openai")
    model = OPENAI_MODEL if provider == "openai" else GEMINI_MODEL
    return provider, model


def call_llm(prompt: str) -> str:
    provider, model_name = _current_llm_config()
    if provider == "openai":
        _ensure_openai()
        try:
            resp = _OPENAI_CLIENT.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return ((resp.choices[0].message.content or "") if resp and resp.choices else "").strip()
        except Exception as e:
            try:
                print(f"⚠️ OpenAI call failed: {e}")
            except Exception:
                pass
            return ""
    # default to gemini
    _ensure_gemini()
    model = genai.GenerativeModel(model_name)
    try:
        resp = model.generate_content(prompt)
        # Prefer resp.text if available; else try candidate parts
        txt = getattr(resp, "text", None)
        if not txt:
            try:
                parts_text: list[str] = []
                for cand in (getattr(resp, "candidates", []) or []):
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in (getattr(content, "parts", []) or []):
                        t = getattr(part, "text", None)
                        if t:
                            parts_text.append(str(t))
                if parts_text:
                    txt = "\n".join(parts_text)
            except Exception:
                txt = None
        # Log safety block if present
        try:
            pf = getattr(resp, "prompt_feedback", None)
            block_reason = getattr(pf, "block_reason", None) if pf else None
            if block_reason:
                print(f"⚠️ Gemini safety block: {block_reason} | ratings={getattr(pf, 'safety_ratings', None)}")
        except Exception:
            pass
        return (txt or "").strip()
    except Exception as e:
        try:
            print(f"⚠️ Gemini call failed: {e}")
        except Exception:
            pass
        return ""


# -----------------------
# 🧰 Utilities
# -----------------------
def ensure_rag_artifacts(mode: str) -> None:
    """Validate local artifacts required for RAG modes.

    Import of retrieval modules is delayed until after this check to avoid
    import-time failures when FAISS indexes are missing.
    """
    if mode not in {"rag", "hybrid", "ontology_cot"}:
        return

    missing = []
    if mode in {"rag", "hybrid", "ontology_cot"}:
        if not os.path.exists("Datasets/indexes/pubmedbert.index"):
            missing.append("Datasets/indexes/pubmedbert.index")
    if mode in {"hybrid", "ontology_cot"}:
        if not os.path.exists("Datasets/indexes/sapbert.index"):
            missing.append("Datasets/indexes/sapbert.index")
    # Shared metadata required by retrieval module
    if not os.path.exists("Datasets/retrieval_corpus/pmid_mapping.json"):
        missing.append("Datasets/retrieval_corpus/pmid_mapping.json")
    if not os.path.exists("Datasets/retrieval_corpus/corpus_subset.jsonl"):
        missing.append("Datasets/retrieval_corpus/corpus_subset.jsonl")

    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required RAG artifact(s): {missing_str}. Run `python embed_corpus.py` first to generate FAISS indexes."
        )


def _mmr_diversify(docs: list[dict], k: int = 12, lambda_param: float = 0.7) -> list[dict]:
    """Simple MMR diversification based on cosine of embedding similarity proxies if present.

    If no vectors are available, falls back to truncation.
    Expects docs may carry 'score' and optionally 'ce_score'. We synthesize a utility score
    and then penalize redundancy by title/abstract token overlap.
    """
    if not docs:
        return []
    k = max(1, min(k, len(docs)))

    def utility(d: dict) -> float:
        return float(d.get("ce_score", 0.0)) * 1.0 + float(d.get("score", 0.0)) * 0.5 + float(d.get("overlap_weighted", 0.0)) * 0.25

    def toks(d: dict) -> set[str]:
        text = f"{d.get('title','')}\n{d.get('abstract','')}"
        text = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
        return set([t for t in text.split() if t])

    selected: list[dict] = []
    remaining = docs[:]
    # Seed by highest utility
    remaining.sort(key=utility, reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e9
        sel_tokens = [toks(d) for d in selected]
        for i, d in enumerate(remaining):
            sim_penalty = 0.0
            dtoks = toks(d)
            for st in sel_tokens:
                inter = len(dtoks & st)
                uni = len(dtoks | st) or 1
                j = inter / uni
                sim_penalty = max(sim_penalty, j)
            score = lambda_param * utility(d) - (1 - lambda_param) * sim_penalty
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected

# -----------------------
# 🚀 MAIN PIPELINE
# -----------------------
def run_pipeline(
    mode: str = "llm_alone",
    output_file: str = "rag_sapbert_outputs.csv",
    top_k: int = 5,
    pre_k: int = 100,
    rows: int = 2,
    verify: bool = False,
    forced_choice: str = "off",  # off | auto | on
    fc_min_k: int = 3,
    fc_max_k: int = 20,
    fc_margin: float = 0.1,
    fc_allow_none: bool = True,
    scaffold: bool = False,
    ce_model: str | None = None,
    force_rag: bool = False,
    sem_weight: float = 0.6,
    bm25_weight: float = 0.2,
    few_shot: bool = False,
    prompt_style: str = "guided",
    no_overlap_rerank: bool = False,
    no_mmr: bool = False,
    no_deterministic: bool = False,
    # Judge/eval options
    use_llm_judge: bool = False,
    judge_provider: str = "auto",
    judge_model: str | None = None,
    max_llm_calls: int = 100,
    verbose_judge: bool = False,
):
    df = pd.read_csv("Datasets/subset_all_mapped.csv")
    # Preserve original row ids so downstream eval can align with gold
    try:
        df["row_id"] = df.index
    except Exception:
        pass
    df = df.sample(n=rows, random_state=42).reset_index(drop=True)
    outputs = []
    # Metrics (computed during loop to enable oracle/candidate coverage)
    oracle_doc_hits = 0
    candidate_cov_hits = 0
    rows_evaluated = 0

    for idx, row in enumerate(df.itertuples(index=False), 0):
        print(f"\n🔍 Processing row {idx + 1} of {len(df)}...")

        note = row.clinical_note
        raw_val = row.matched_hpo_terms

        matched = []
        if pd.notna(raw_val):
            s = str(raw_val).strip()
            try:
                # Prefer JSON if it looks like JSON
                if s.startswith("[") or s.startswith("{"):
                    try:
                        matched = json.loads(s)
                    except Exception:
                        # Normalize JSON literals to Python and retry with ast
                        s_py = s.replace("null", "None").replace("true", "True").replace("false", "False")
                        matched = ast.literal_eval(s_py)
                else:
                    matched = ast.literal_eval(s)
            except Exception:
                try:
                    # Last-ditch cleanup of smart quotes
                    s2 = s.replace("“", '"').replace("”", '"').replace("’", "'")
                    matched = ast.literal_eval(s2)
                except Exception:
                    print(f"⚠️ matched_hpo_terms parse failed for row {idx}; falling back to empty list")
                    matched = []

        # Normalize to list of term strings
        hpo_terms: list[str] = []
        for term in matched:
            if isinstance(term, (list, tuple)) and len(term) >= 1:
                hpo_terms.append(term[0])
            elif isinstance(term, dict):
                t = term.get("term") or term.get("name") or term.get("hpo") or ""
                if t:
                    hpo_terms.append(t)
            elif isinstance(term, str):
                hpo_terms.append(term)
        hpo_terms = [t for t in hpo_terms if isinstance(t, str) and t.strip()]

        # Verbose: basic case info
        try:
            _note_snip = str(note)[:140]
            _note_snip = _note_snip.replace("\n", " ")
            print(f"Case note snippet: {_note_snip}...")
            print(f"HPO terms ({len(hpo_terms)}): {hpo_terms[:20]}")
        except Exception:
            pass

        try:
            if mode == "llm_alone":
                if prompt_style == "zero_shot_top5":
                    prompt = build_prompt_llm_alone_zero_shot_top5(note, hpo_terms)
                elif scaffold:
                    prompt = build_prompt_ontology_cot(note, hpo_terms, [])
                else:
                    prompt = build_prompt_llm_alone(note, hpo_terms)

            elif mode == "rag":
                ensure_rag_artifacts(mode)
                # Lazy import to avoid import-time failures if indexes are missing
                from corpus_retrieval_patched import retrieve_pubmedbert  # noqa: WPS433
                # Optional helpers from UMLS module (graceful fallback)
                try:
                    from rag_scispacy_umls import (
                        augment_with_idf_weighted_overlap,
                        rerank_with_overlap,
                        rerank_with_cross_encoder,
                        extract_disease_candidates_from_docs,
                        extract_disease_candidates_from_note,
                        expand_terms_with_umls_synonyms,
                        extract_gene_like_tokens,
                        score_candidates,
                        build_literature_block,
                    )
                except Exception:
                    augment_with_idf_weighted_overlap = None
                    rerank_with_overlap = None
                    rerank_with_cross_encoder = None
                    extract_disease_candidates_from_docs = None
                    extract_disease_candidates_from_note = None
                    expand_terms_with_umls_synonyms = None
                    extract_gene_like_tokens = None
                    score_candidates = None
                    build_literature_block = None

                # Multi-query retrieval: note, HPO list, combined, expanded synonyms, and gene-augmented
                terms_for_query = hpo_terms
                try:
                    expanded_terms = (
                        expand_terms_with_umls_synonyms(terms_for_query, max_synonyms_per=2)  # type: ignore[arg-type]
                        if expand_terms_with_umls_synonyms is not None else terms_for_query
                    )
                except Exception:
                    expanded_terms = terms_for_query
                try:
                    genes = list(extract_gene_like_tokens(note)) if extract_gene_like_tokens is not None else []
                except Exception:
                    genes = []

                from itertools import chain as _chain
                queries = [
                    note,
                    ", ".join(terms_for_query),
                    note + "\n" + ", ".join(terms_for_query),
                    ", ".join(expanded_terms[:30]),
                    (note + "\nGENES: " + ", ".join(genes)) if genes else note,
                ]
                pools = []
                for q in queries:
                    try:
                        pools.append(retrieve_pubmedbert(q, top_k=50))
                    except Exception:
                        continue
                # Merge by PMID preserving best semantic score
                merged: dict[str, dict] = {}
                for doc in _chain.from_iterable(pools):
                    key = str(doc.get("pmid"))
                    if key not in merged:
                        merged[key] = dict(doc)
                    else:
                        merged[key]["score"] = max(float(merged[key].get("score", 0.0)), float(doc.get("score", 0.0)))
                docs = list(merged.values())

                # Verbose: retrieval summary
                try:
                    print(f"Retrieved {len(docs)} docs (rag/pubmedbert path)")
                    if docs:
                        preview = []
                        for d in docs[:5]:
                            preview.append(f"PMID {d.get('pmid')} score={d.get('score'):.4f} ce={d.get('ce_score', 0.0)} ov={d.get('overlap_weighted', 0.0)}")
                        print("Top docs:", "; ".join(preview))
                except Exception:
                    pass

                # Annotate and rerank
                if augment_with_idf_weighted_overlap is not None and not no_overlap_rerank:
                    docs = augment_with_idf_weighted_overlap(docs, terms_for_query)  # type: ignore[arg-type]
                if rerank_with_overlap is not None and not no_overlap_rerank:
                    docs = rerank_with_overlap(docs, expanded_terms or terms_for_query, threshold=1, top_k=max(30, top_k))  # type: ignore[arg-type]
                if rerank_with_cross_encoder is not None:
                    # Allow overriding model name from CLI
                    if ce_model:
                        docs = rerank_with_cross_encoder(note, docs, top_k=max(12, top_k), model_name=ce_model)
                    else:
                        docs = rerank_with_cross_encoder(note, docs, top_k=max(12, top_k))

                # Optional MMR diversification
                if not no_mmr:
                    try:
                        docs = _mmr_diversify(docs, k=max(12, top_k), lambda_param=0.7)
                    except Exception:
                        docs = docs[:max(12, top_k)]
                else:
                    docs = docs[:max(12, top_k)]

                # Candidate mining and deterministic decision layer
                candidates: list[str] = []
                if extract_disease_candidates_from_docs is not None:
                    try:
                        candidates = extract_disease_candidates_from_docs(docs, max_candidates=30)
                    except Exception:
                        candidates = []
                if extract_disease_candidates_from_note is not None:
                    try:
                        note_cands = extract_disease_candidates_from_note(note, max_candidates=10)
                        seen = set(candidates)
                        for c in note_cands:
                            if c not in seen:
                                candidates.append(c)
                                seen.add(c)
                    except Exception:
                        pass

                # Verbose: candidates mined
                try:
                    print(f"Candidates mined ({len(candidates)}): {candidates[:15]}")
                except Exception:
                    pass

                final_choice = None
                top_support: list[str] = []
                scored = []
                if score_candidates is not None and candidates and not no_deterministic:
                    try:
                        scored = score_candidates(candidates, docs, note=note, note_hpo_terms=terms_for_query)
                        if scored:
                            # Only use deterministic if score is very high
                            top_score = float(scored[0].get("score", 0.0))
                            if top_score >= 3.0:  # High threshold for deterministic choice
                                final_choice = scored[0]["canonical"]
                                top_support = scored[0]["support_pmids"]
                        # Verbose: scored candidates
                        try:
                            if scored:
                                print("Top scored candidates:")
                                for s in scored[:5]:
                                    print(f"  - {s.get('canonical')} | score={s.get('score')} | pmids={s.get('support_pmids', [])[:3]}")
                            else:
                                print("No scored candidates (empty)")
                        except Exception:
                            pass
                    except Exception:
                        final_choice = None

                strategy = "decision"
                if final_choice and build_literature_block is not None and not no_deterministic:
                    literature = build_literature_block(docs, include_overlap=True)
                    few_shot_block = ("Example:\nFinal diagnosis: Rett syndrome\nAliases: RTTS\nUMLS_CUI: C0035372\nPMIDs: 20301510, 32623421\nRationale: Key features X,Y,Z; supported by PMIDs.\n\n" if few_shot else "")
                    prompt = (
                        f"You are a clinical diagnosis assistant.\n"
                        f"{few_shot_block}"
                        f"Output format (exact):\n"
                        f"Final diagnosis: {final_choice}\n"
                        f"Rationale: <2-4 sentences citing strongest PMIDs>\n\n"
                        f"Retrieved Literature (with overlaps):\n{literature}"
                    )
                    strategy = "deterministic"
                else:
                    use_forced = False
                    if forced_choice == "on" and candidates:
                        use_forced = True
                    elif forced_choice == "auto" and scored:
                        # auto-gate by candidate size and score margin
                        if fc_min_k <= len(scored) <= fc_max_k:
                            s0 = float(scored[0].get("score", 0.0))
                            s1 = float(scored[1].get("score", 0.0)) if len(scored) > 1 else 0.0
                            if (s0 - s1) >= fc_margin:
                                use_forced = True
                    if use_forced:
                        # Filter generic/non-specific candidates and canonicalize to dataset labels where possible
                        allowed = []
                        generic_bans = {"intellectual disability", "congenital abnormality", "disease", "developmental delay"}
                        seen_norm = set()
                        for c in candidates:
                            cc = canonicalize_disease_name(c) if canonicalize_disease_name is not None else c
                            # Try mapping to dataset label
                            try:
                                from rag_scispacy_umls import map_to_dataset_label  # type: ignore
                            except Exception:
                                map_to_dataset_label = None  # type: ignore
                            mapped = map_to_dataset_label(cc) if map_to_dataset_label is not None else None
                            label_like = mapped or cc
                            cn = normalize_text(str(label_like))
                            if not cn or cn in generic_bans:
                                continue
                            if cn in seen_norm:
                                continue
                            allowed.append(str(label_like))
                            seen_norm.add(cn)
                        if not allowed:
                            allowed = list(candidates)
                        if fc_allow_none and "None of the above" not in allowed:
                            allowed.append("None of the above")
                        prompt = build_prompt_rag_forced_choice(note, terms_for_query, docs, allowed)
                        strategy = "forced_choice"
                        # Legacy forced-choice reconsideration: if model returns 'None of the above', try once more
                        if os.getenv("LEGACY_FC", "0") == "1":
                            _fc_out = call_llm(prompt)
                            _fc_choice = extract_final_diagnosis(_fc_out)
                            if normalize_text(_fc_choice) == normalize_text("None of the above"):
                                recon_prompt = build_prompt_rag_reconsider(note, terms_for_query, docs, _fc_out, allowed)
                                prompt = recon_prompt
                        try:
                            print(f"Strategy: forced_choice | allowed ({len(allowed)}): {allowed[:10]}")
                        except Exception:
                            pass
                    else:
                        if scaffold:
                            prompt = build_prompt_ontology_cot(note, terms_for_query, docs)
                        else:
                            base = build_prompt_rag_decision(note, terms_for_query, docs)
                            if few_shot:
                                prompt = (
                                    "Example:\nFinal diagnosis: Rett syndrome\nAliases: RTTS\nUMLS_CUI: C0035372\nPMIDs: 20301510, 32623421\nRationale: Key features X,Y,Z; supported by PMIDs.\n\n"
                                    + base
                                )
                            else:
                                prompt = base
                        strategy = "decision"
                        try:
                            print(f"Strategy: {strategy} | scaffold={scaffold} few_shot={few_shot}")
                        except Exception:
                            pass

                # Row-level oracle and coverage metrics
                gold_label = getattr(row, "disease", "")
                if canonicalize_disease_name is not None:
                    gold_label = canonicalize_disease_name(str(gold_label))
                goldn = normalize_text(str(gold_label))
                try:
                    doc_text = "\n".join([f"{d.get('title','')}\n{d.get('abstract','')}" for d in docs])
                    if goldn and normalize_text(doc_text).find(goldn) != -1:
                        oracle_doc_hits += 1
                except Exception:
                    pass
                try:
                    norm_cands = []
                    for c in candidates:
                        cc = canonicalize_disease_name(c) if canonicalize_disease_name is not None else c
                        norm_cands.append(normalize_text(str(cc)))
                    if goldn and any((goldn == c or goldn in c or c in goldn) for c in norm_cands):
                        candidate_cov_hits += 1
                except Exception:
                    pass
                rows_evaluated += 1

            elif mode == "hybrid":
                ensure_rag_artifacts(mode)
                from corpus_retrieval_patched import retrieve_hybrid  # noqa: WPS433

                # Optional helpers from UMLS module (graceful fallback)
                try:
                    from rag_scispacy_umls import (
                        augment_with_idf_weighted_overlap,
                        rerank_with_overlap,
                        rerank_with_cross_encoder,
                        extract_disease_candidates_from_docs,
                        extract_disease_candidates_from_note,
                        expand_terms_with_umls_synonyms,
                        extract_gene_like_tokens,
                        score_candidates,
                        build_literature_block,
                    )
                except Exception:
                    augment_with_idf_weighted_overlap = None
                    rerank_with_overlap = None
                    rerank_with_cross_encoder = None
                    extract_disease_candidates_from_docs = None
                    extract_disease_candidates_from_note = None
                    expand_terms_with_umls_synonyms = None
                    extract_gene_like_tokens = None
                    score_candidates = None
                    build_literature_block = None

                terms_for_query = hpo_terms
                try:
                    expanded_terms = (
                        expand_terms_with_umls_synonyms(terms_for_query, max_synonyms_per=2)  # type: ignore[arg-type]
                        if expand_terms_with_umls_synonyms is not None else terms_for_query
                    )
                except Exception:
                    expanded_terms = terms_for_query
                try:
                    genes = list(extract_gene_like_tokens(note)) if extract_gene_like_tokens is not None else []
                except Exception:
                    genes = []

                from itertools import chain as _chain
                queries = [
                    note,
                    note + "\n" + ", ".join(terms_for_query),
                    ", ".join(expanded_terms[:30]),
                    (note + "\nGENES: " + ", ".join(genes)) if genes else note,
                ]
                pools = []
                for q in queries:
                    try:
                        pools.append(retrieve_hybrid(q, terms_for_query, top_k=max(10, pre_k), sem_weight=sem_weight, bm25_weight=bm25_weight, fusion_mode=("legacy" if bm25_weight <= 0.0 else "normalized")))
                    except Exception:
                        continue
                merged: dict[str, dict] = {}
                for doc in _chain.from_iterable(pools):
                    key = str(doc.get("pmid"))
                    if key not in merged:
                        merged[key] = dict(doc)
                    else:
                        merged[key]["score"] = max(float(merged[key].get("score", 0.0)), float(doc.get("score", 0.0)))
                docs = list(merged.values())

                # Verbose: retrieval summary (hybrid)
                try:
                    print(f"Retrieved {len(docs)} docs (hybrid path)")
                    if docs:
                        preview = []
                        for d in docs[:5]:
                            preview.append(f"PMID {d.get('pmid')} score={d.get('score', 0.0)} ce={d.get('ce_score', 0.0)} ov={d.get('overlap_weighted', 0.0)}")
                        print("Top docs:", "; ".join(preview))
                except Exception:
                    pass

                if augment_with_idf_weighted_overlap is not None and not no_overlap_rerank:
                    docs = augment_with_idf_weighted_overlap(docs, terms_for_query)  # type: ignore[arg-type]
                if rerank_with_overlap is not None and not no_overlap_rerank:
                    docs = rerank_with_overlap(docs, expanded_terms or terms_for_query, threshold=1, top_k=max(30, top_k))  # type: ignore[arg-type]
                if rerank_with_cross_encoder is not None:
                    if ce_model:
                        docs = rerank_with_cross_encoder(note, docs, top_k=max(12, top_k), model_name=ce_model)
                    else:
                        docs = rerank_with_cross_encoder(note, docs, top_k=max(12, top_k))
                if not no_mmr:
                    try:
                        docs = _mmr_diversify(docs, k=max(12, top_k), lambda_param=0.7)
                    except Exception:
                        docs = docs[:max(12, top_k)]
                else:
                    docs = docs[:max(12, top_k)]

                candidates: list[str] = []
                if extract_disease_candidates_from_docs is not None:
                    try:
                        candidates = extract_disease_candidates_from_docs(docs, max_candidates=30)
                    except Exception:
                        candidates = []
                if extract_disease_candidates_from_note is not None:
                    try:
                        note_cands = extract_disease_candidates_from_note(note, max_candidates=10)
                        seen = set(candidates)
                        for c in note_cands:
                            if c not in seen:
                                candidates.append(c)
                                seen.add(c)
                    except Exception:
                        pass

                # Verbose: candidates mined (hybrid)
                try:
                    print(f"Candidates mined ({len(candidates)}): {candidates[:15]}")
                except Exception:
                    pass

                final_choice = None
                top_support: list[str] = []
                scored = []
                if score_candidates is not None and candidates and not no_deterministic:
                    try:
                        scored = score_candidates(candidates, docs, note=note, note_hpo_terms=terms_for_query)
                        if scored:
                            # Only use deterministic if score is very high
                            top_score = float(scored[0].get("score", 0.0))
                            if top_score >= 3.0:  # High threshold for deterministic choice
                                final_choice = scored[0]["canonical"]
                                top_support = scored[0]["support_pmids"]
                        # Verbose: scored candidates (hybrid)
                        try:
                            if scored:
                                print("Top scored candidates:")
                                for s in scored[:5]:
                                    print(f"  - {s.get('canonical')} | score={s.get('score')} | pmids={s.get('support_pmids', [])[:3]}")
                            else:
                                print("No scored candidates (empty)")
                        except Exception:
                            pass
                    except Exception:
                        final_choice = None

                strategy = "decision"
                if final_choice and build_literature_block is not None and not no_deterministic:
                    literature = build_literature_block(docs, include_overlap=True)
                    few_shot_block = ("Example:\nFinal diagnosis: Rett syndrome\nAliases: RTTS\nUMLS_CUI: C0035372\nPMIDs: 20301510, 32623421\nRationale: Key features X,Y,Z; supported by PMIDs.\n\n" if few_shot else "")
                    prompt = (
                        f"You are a clinical diagnosis assistant.\n"
                        f"{few_shot_block}"
                        f"Output format (exact):\n"
                        f"Final diagnosis: {final_choice}\n"
                        f"Rationale: <2-4 sentences citing strongest PMIDs>\n\n"
                        f"Retrieved Literature (with overlaps):\n{literature}"
                    )
                    strategy = "deterministic"
                else:
                    use_forced = False
                    if forced_choice == "on" and candidates:
                        use_forced = True
                    elif forced_choice == "auto" and scored:
                        if fc_min_k <= len(scored) <= fc_max_k:
                            s0 = float(scored[0].get("score", 0.0))
                            s1 = float(scored[1].get("score", 0.0)) if len(scored) > 1 else 0.0
                            if (s0 - s1) >= fc_margin:
                                use_forced = True
                    if use_forced:
                        allowed = list(candidates)
                        if fc_allow_none and "None of the above" not in allowed:
                            allowed.append("None of the above")
                        prompt = build_prompt_rag_forced_choice(note, terms_for_query, docs, allowed)
                        strategy = "forced_choice"
                        try:
                            print(f"Strategy: forced_choice | allowed ({len(allowed)}): {allowed[:10]}")
                        except Exception:
                            pass
                    else:
                        if scaffold:
                            prompt = build_prompt_ontology_cot(note, terms_for_query, docs)
                        else:
                            base = build_prompt_rag_decision(note, terms_for_query, docs)
                            if few_shot:
                                prompt = (
                                    "Example:\nFinal diagnosis: Rett syndrome\nAliases: RTTS\nUMLS_CUI: C0035372\nPMIDs: 20301510, 32623421\nRationale: Key features X,Y,Z; supported by PMIDs.\n\n"
                                    + base
                                )
                            else:
                                prompt = base
                        strategy = "decision"
                        try:
                            print(f"Strategy: {strategy} | scaffold={scaffold} few_shot={few_shot}")
                        except Exception:
                            pass

                gold_label = getattr(row, "disease", "")
                if canonicalize_disease_name is not None:
                    gold_label = canonicalize_disease_name(str(gold_label))
                goldn = normalize_text(str(gold_label))
                try:
                    doc_text = "\n".join([f"{d.get('title','')}\n{d.get('abstract','')}" for d in docs])
                    if goldn and normalize_text(doc_text).find(goldn) != -1:
                        oracle_doc_hits += 1
                except Exception:
                    pass
                try:
                    norm_cands = []
                    for c in candidates:
                        cc = canonicalize_disease_name(c) if canonicalize_disease_name is not None else c
                        norm_cands.append(normalize_text(str(cc)))
                    if goldn and any((goldn == c or goldn in c or c in goldn) for c in norm_cands):
                        candidate_cov_hits += 1
                except Exception:
                    pass
                rows_evaluated += 1

            elif mode == "ontology_cot":
                ensure_rag_artifacts(mode)
                from corpus_retrieval_patched import retrieve_hybrid  # noqa: WPS433
                docs = retrieve_hybrid(note, hpo_terms, top_k=max(top_k, pre_k), sem_weight=sem_weight, bm25_weight=bm25_weight, fusion_mode=("legacy" if bm25_weight <= 0.0 else "normalized"))[:top_k]
                prompt = build_prompt_ontology_cot(note, hpo_terms, docs)

            elif mode == "scispacy_umls":
                ensure_rag_artifacts("rag")
                
                # Import functions at top level to avoid scoping issues
                try:
                    from rag_scispacy_umls import (
                        extract_symptom_terms_with_umls,
                        rerank_with_overlap,
                        extract_disease_candidates_from_docs,
                        extract_disease_candidates_from_note,
                        augment_with_idf_weighted_overlap,
                        rerank_with_cross_encoder,
                        score_candidates,
                        expand_terms_with_umls_synonyms,
                        extract_gene_like_tokens,
                    )
                except Exception:
                    extract_symptom_terms_with_umls = None
                    rerank_with_overlap = None
                    extract_disease_candidates_from_docs = None
                    extract_disease_candidates_from_note = None
                    augment_with_idf_weighted_overlap = None
                    rerank_with_cross_encoder = None
                    score_candidates = None
                    expand_terms_with_umls_synonyms = None
                    extract_gene_like_tokens = None
                
                # CONDITIONAL RAG: Try LLM-alone first, use RAG only if uncertain
                
                # Step 1: LLM-alone reasoning
                llm_alone_prompt = build_prompt_llm_alone(note, hpo_terms)
                llm_alone_response = call_llm(llm_alone_prompt)
                llm_alone_diagnosis = extract_final_diagnosis(llm_alone_response)
                
                # Step 2: Check confidence/uncertainty indicators
                uncertainty_indicators = [
                    "uncertain", "unclear", "difficult to determine", "multiple possibilities",
                    "differential", "consider", "possible", "likely", "suggests", "may be",
                    "cannot rule out", "further testing", "additional workup"
                ]
                
                is_uncertain = force_rag or any(indicator in llm_alone_response.lower() for indicator in uncertainty_indicators)
                is_empty_or_vague = not llm_alone_diagnosis or len(llm_alone_diagnosis.strip()) < 5
                
                # Branch: only run RAG pipeline if uncertain; otherwise, use confident LLM output
                if is_uncertain or is_empty_or_vague:
                    print(f"🔍 LLM uncertain - activating RAG pipeline...")

                    extracted_terms = []
                    if extract_symptom_terms_with_umls is not None:
                        extracted_terms = extract_symptom_terms_with_umls(note)
                    terms_for_query = extracted_terms or hpo_terms
                    # Expand terms with UMLS synonyms and add gene-like tokens
                    try:
                        expanded = expand_terms_with_umls_synonyms(terms_for_query, max_synonyms_per=2)
                    except Exception:
                        expanded = terms_for_query
                    try:
                        genes = list(extract_gene_like_tokens(note))
                    except Exception:
                        genes = []

                    # Use PubMedBERT + BM25 retrieval with SciSpaCy extracted terms
                    from corpus_retrieval_patched import retrieve_pubmedbert, retrieve_bm25  # noqa: WPS433

                    # Multi-query PubMedBERT retrieval
                    from itertools import chain
                    queries = [
                        note,
                        ", ".join(terms_for_query),
                        ", ".join(expanded[:30]) if expanded else ", ".join(terms_for_query),
                    ]
                    pools = []
                    for q in queries:
                        if q.strip():
                            pools.append(retrieve_pubmedbert(q, top_k=max(20, pre_k // 2)))
                            # Also add BM25 hits into the pool
                            try:
                                pools.append(retrieve_bm25(q, top_k=max(20, pre_k // 2)))
                            except Exception:
                                pass

                    # Merge by PMID preserving best score
                    merged: dict[str, dict] = {}
                    for doc in chain.from_iterable(pools):
                        key = str(doc.get("pmid"))
                        if key not in merged:
                            merged[key] = dict(doc)
                        else:
                            # Preserve highest available similarity-like score across channels
                            s_old = float(merged[key].get("score", 0.0))
                            s_new = float(doc.get("score", doc.get("bm25", 0.0)))
                            merged[key]["score"] = max(s_old, s_new)
                    base_docs: list[dict] = list(merged.values())

                    # Apply overlap filter + rerank
                    docs = base_docs
                    if augment_with_idf_weighted_overlap is not None and not no_overlap_rerank:
                        docs = augment_with_idf_weighted_overlap(docs, terms_for_query)
                    if rerank_with_overlap is not None and not no_overlap_rerank:
                        # use expanded terms for overlap
                        docs = rerank_with_overlap(docs, expanded or terms_for_query, threshold=2, top_k=max(30, top_k))
                    if rerank_with_cross_encoder is not None:
                        docs = rerank_with_cross_encoder(note, docs, top_k=max(12, top_k))

                    # Build forced-choice candidate list
                    candidates: list[str] = []
                    if extract_disease_candidates_from_docs is not None:
                        try:
                            # Disable UMLS in candidate extractor to reduce noise
                            candidates = extract_disease_candidates_from_docs(docs, max_candidates=30, use_umls=False)
                        except Exception:
                            candidates = []
                    # also mine from note
                    if extract_disease_candidates_from_note is not None:
                        try:
                            note_cands = extract_disease_candidates_from_note(note, max_candidates=10)
                            # Merge
                            seen = set(candidates)
                            for c in note_cands:
                                if c not in seen:
                                    candidates.append(c)
                                    seen.add(c)
                        except Exception:
                            pass

                    # Deterministic decision (disabled unless very strong)
                    final_choice = None
                    top_support = []
                    # if candidates:
                    #     try:
                    #         scored = score_candidates(candidates, docs, note=note, note_hpo_terms=terms_for_query)
                    #         if scored:
                    #             top_score = float(scored[0].get("score", 0.0))
                    #             if top_score >= 3.0:
                    #                 final_choice = scored[0]["canonical"]
                    #                 top_support = scored[0]["support_pmids"]
                    #     except Exception:
                    #         final_choice = None

                    if final_choice:
                        from rag_scispacy_umls import build_literature_block
                        literature = build_literature_block(docs, include_overlap=True)
                        support_str = ", ".join(top_support[:5]) if top_support else ""
                        prompt = (
                            f"You are a clinical diagnosis assistant.\n"
                            f"Output format (exact):\n"
                            f"Final diagnosis: {final_choice}\n"
                            f"Rationale: <2-4 sentences citing strongest PMIDs>\n\n"
                            f"Retrieved Literature (with overlaps):\n{literature}"
                        )
                    else:
                        if candidates:
                            prompt = build_prompt_rag_forced_choice(note, terms_for_query, docs, candidates)
                        else:
                            if scaffold:
                                prompt = build_prompt_ontology_cot(note, terms_for_query, docs)
                            else:
                                prompt = build_prompt_rag_decision(note, terms_for_query, docs)

                    # Row-level oracle and coverage metrics
                    gold_label = getattr(row, "disease", "")
                    if canonicalize_disease_name is not None:
                        gold_label = canonicalize_disease_name(str(gold_label))
                    goldn = normalize_text(str(gold_label))
                    try:
                        doc_text = "\n".join([f"{d.get('title','')}\n{d.get('abstract','')}" for d in docs])
                        if goldn and normalize_text(doc_text).find(goldn) != -1:
                            oracle_doc_hits += 1
                    except Exception:
                        pass
                    try:
                        norm_cands = []
                        for c in candidates:
                            cc = canonicalize_disease_name(c) if canonicalize_disease_name is not None else c
                            norm_cands.append(normalize_text(str(cc)))
                        if goldn and any((goldn == c or goldn in c or c in goldn) for c in norm_cands):
                            candidate_cov_hits += 1
                        else:
                            if rows_evaluated <= 2 and goldn and candidates:
                                print(f"\n⚠️ Gold '{gold_label}' not in candidates for row {rows_evaluated}")
                                print(f"   Top 5 candidates: {candidates[:5]}")
                    except Exception:
                        pass
                    rows_evaluated += 1
                    strategy = "decision"
                else:
                    # LLM was confident - skip RAG pipeline and keep earlier output
                    print(f"✅ LLM confident - using diagnosis: {llm_alone_diagnosis}")
                    strategy = "llm_confident"
                    prompt = None
                    docs = []
                    candidates = []
                    rows_evaluated += 1

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Handle conditional LLM call
            if locals().get('strategy') == 'llm_confident':
                # Use the original LLM-alone response
                response = llm_alone_response
                print(f"Using confident LLM response (no additional call)")
            else:
                # Verbose: LLM call details
                try:
                    _prov, _model = _current_llm_config()
                    print(f"LLM call → provider={_prov} model={_model} | strategy={locals().get('strategy','decision')} | prompt_chars={len(prompt)}")
                    print("Prompt head:\n" + str(prompt)[:240])
                except Exception:
                    pass

                response = call_llm(prompt)

            try:
                if not response:
                    print("LLM response is empty")
                else:
                    print("LLM response head:\n" + str(response)[:240])
            except Exception:
                pass

            # Optional verify/revise pass
            verified_output = response
            if verify:
                try:
                    verify_prompt = build_prompt_verify(note, hpo_terms, locals().get("docs", []), response)
                    verified_output = call_llm(verify_prompt)
                    try:
                        print("Verify pass used. Verified output head:\n" + str(verified_output)[:240])
                    except Exception:
                        pass
                except Exception as _verr:
                    verified_output = response

            # Extract the actual diagnosis from the LLM output
            # Use final_choice if it was set (for deterministic scispacy_umls mode)
            # Otherwise extract from the LLM response
            if 'final_choice' in locals() and final_choice:
                extracted_diagnosis = final_choice
            else:
                extracted_diagnosis = extract_final_diagnosis(verified_output)
                # Clean up common template artifacts
                if extracted_diagnosis in ["<DISEASE>", "disease", "<one item copied verbatim from Allowed diagnoses>"]:
                    extracted_diagnosis = extract_final_diagnosis(response)
                    if extracted_diagnosis in ["<DISEASE>", "disease", "<one item copied verbatim from Allowed diagnoses>"]:
                        extracted_diagnosis = ""

            try:
                print(f"Extracted final diagnosis: '{extracted_diagnosis or '<EMPTY>'}'")
            except Exception:
                pass

            ranked = extract_ranked_diagnoses(verified_output)
            ranked_serialized = serialize_ranked_diagnoses(ranked)

            outputs.append({
                "row_id": getattr(row, "row_id", None),
                "clinical_note": note,
                "hpo_terms": hpo_terms,
                "prompt": prompt,
                "llm_output": verified_output,
                "llm_output_initial": response,
                "final_diagnosis": extracted_diagnosis,
                "strategy": locals().get("strategy", "decision"),
                "prompt_style": prompt_style if mode == "llm_alone" else "",
                "ranked_diagnoses": ranked_serialized,
            })

        except Exception as e:
            print(f"⚠️ Error at row {idx}: {e}")

    pd.DataFrame(outputs).to_csv(output_file, index=False)

    # Compute simple end-of-run accuracy against ground-truth `disease` for processed rows
    try:
        preds = []
        labels = []
        for out, lab in zip([o.get("llm_output", "") for o in outputs], df["disease"].tolist()):
            pred = normalize_text(extract_final_diagnosis(out))
            labn = normalize_text(str(lab))
            preds.append(pred)
            labels.append(labn)

        total = len(preds)
        strict_hits = sum(1 for p, l in zip(preds, labels) if p and p == l)

        def _toks(s: str) -> set:
            return set([t for t in s.split() if t])

        def _jacc(a: set, b: set) -> float:
            return (len(a & b) / len(a | b)) if (a or b) else 0.0

        soft_hits = 0
        for p, l in zip(preds, labels):
            if not p or not l:
                continue
            if p in l or l in p or _jacc(_toks(p), _toks(l)) >= 0.5:
                soft_hits += 1

        print(f"\n✅ Completed {len(outputs)} rows → saved to {output_file}")
        print(
            f"Accuracy (strict): {strict_hits}/{total} = {strict_hits/total*100:.2f}%\n"
            f"Accuracy (soft  ): {soft_hits}/{total} = {soft_hits/total*100:.2f}%"
        )
        
        # Diagnostic: Show first 3 mismatches to understand the problem
        if strict_hits < total:
            print("\n🔍 Diagnostic - First 3 mismatches (normalized):")
            shown = 0
            for i, (p, l, out) in enumerate(zip(preds, labels, [o.get("llm_output", "") for o in outputs])):
                if p != l and shown < 3:
                    raw_pred = extract_final_diagnosis(out)
                    raw_gold = df.iloc[i]["disease"]
                    print(f"  Row {i+1}:")
                    print(f"    Gold (raw): '{raw_gold}'")
                    print(f"    Gold (norm): '{l}'")
                    print(f"    Pred (raw): '{raw_pred}'")
                    print(f"    Pred (norm): '{p}'")
                    shown += 1
        if rows_evaluated > 0:
            print(
                f"Oracle doc@K: {oracle_doc_hits}/{rows_evaluated} = {oracle_doc_hits/max(1,rows_evaluated)*100:.2f}%\n"
                f"Candidate coverage: {candidate_cov_hits}/{rows_evaluated} = {candidate_cov_hits/max(1,rows_evaluated)*100:.2f}%"
            )
        # Optional: run LLM judge on outputs to print adjusted accuracy (only increases)
        if use_llm_judge and _eval_compute_accuracy is not None:
            try:
                # Default judge model: prefer GEMINI if available
                jm = judge_model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                _eval_compute_accuracy(
                    results_csv=output_file,
                    gold_csv="Datasets/subset_all_mapped.csv",
                    nrows=rows,
                    use_llm_judge=True,
                    judge_model=jm,
                    max_llm_calls=max_llm_calls,
                    verbose_judge=verbose_judge,
                    judge_provider=judge_provider,
                )
            except Exception as _je:
                print(f"⚠️ LLM judge evaluation skipped: {_je}")
    except Exception as acc_err:
        print(f"⚠️ Accuracy computation failed: {acc_err}")


# 🏁 Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM diagnosis pipeline locally")
    parser.add_argument("--mode", choices=["llm_alone", "hybrid", "scispacy_umls"], default="llm_alone", help="Pipeline mode")
    parser.add_argument("--output_file", default="outputs.csv", help="Path to write CSV outputs")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve for RAG modes (final top-k to send to LLM)")
    parser.add_argument("--pre_k", type=int, default=100, help="Pre-rerank pool size before cross-encoder and final cut")
    parser.add_argument("--rows", type=int, default=2, help="Number of dataset rows to process")
    parser.add_argument("--verify", action="store_true", help="Enable a second verify/revise pass before finalizing output")
    parser.add_argument(
        "--prompt_style",
        choices=["guided", "zero_shot_top5"],
        default="guided",
        help="Prompt template to use in llm-alone mode",
    )
    parser.add_argument("--forced_choice", choices=["off", "auto", "on"], default="auto", help="Use forced-choice prompt gating: off|auto|on")
    parser.add_argument("--fc_min_k", type=int, default=3, help="Min candidate size to allow forced-choice (auto mode)")
    parser.add_argument("--fc_max_k", type=int, default=20, help="Max candidate size to allow forced-choice (auto mode)")
    parser.add_argument("--fc_margin", type=float, default=0.1, help="Min score margin (top1-top2) to allow forced-choice (auto mode)")
    parser.add_argument("--fc_allow_none", action="store_true", help="Include 'None of the above' in forced-choice list")
    parser.add_argument("--scaffold", action="store_true", help="Use ontology scaffold style prompting during decision (rag/hybrid)")
    parser.add_argument("--ce_model", default=None, help="Cross-encoder model name for reranking (e.g., 'bge-reranker-base')")
    parser.add_argument("--force_rag", action="store_true", help="Always run retrieval+rationale even if LLM-alone looks confident")
    parser.add_argument("--sem_weight", type=float, default=0.6, help="Hybrid fusion weight for semantic (PubMedBERT)")
    parser.add_argument("--bm25_weight", type=float, default=0.2, help="Hybrid fusion weight for lexical BM25")
    parser.add_argument("--few_shot", action="store_true", help="Prepend one concise example to decision prompts for stability")
    parser.add_argument("--no_deterministic", action="store_true", help="Disable deterministic decision layer in hybrid/scispacy modes")
    parser.add_argument("--no_mmr", action="store_true", help="Disable MMR diversification in hybrid mode")
    parser.add_argument("--no_overlap_rerank", action="store_true", help="Disable overlap-based rerank in hybrid/scispacy modes")
    # Judge/eval flags (prints adjusted accuracy after run; does not change CSV labels)
    parser.add_argument("--use_llm_judge", action="store_true", help="After run, print adjusted accuracy using LLM judge")
    parser.add_argument("--judge_provider", choices=["auto", "openai", "gemini"], default="auto", help="Provider for LLM judge")
    parser.add_argument("--judge_model", default=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), help="Model for LLM judge")
    parser.add_argument("--max_llm_calls", type=int, default=100, help="Max number of LLM judge calls")
    parser.add_argument("--verbose_judge", action="store_true", help="Verbose per-case judge logging")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        output_file=args.output_file,
        top_k=args.top_k,
        pre_k=args.pre_k,
        rows=args.rows,
        verify=args.verify,
        forced_choice=args.forced_choice,
        fc_min_k=args.fc_min_k,
        fc_max_k=args.fc_max_k,
        fc_margin=args.fc_margin,
        fc_allow_none=args.fc_allow_none,
        scaffold=args.scaffold,
        ce_model=args.ce_model,
        few_shot=args.few_shot,
        force_rag=args.force_rag,
        sem_weight=args.sem_weight,
        bm25_weight=args.bm25_weight,
        prompt_style=args.prompt_style,
        no_overlap_rerank=args.no_overlap_rerank,
        no_mmr=args.no_mmr,
        no_deterministic=args.no_deterministic,
        use_llm_judge=args.use_llm_judge,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        max_llm_calls=args.max_llm_calls,
        verbose_judge=args.verbose_judge,
    )
