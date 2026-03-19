import re
import os
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _CROSS_ENCODER_AVAILABLE = True
except Exception:
    _CROSS_ENCODER_AVAILABLE = False

# Optional SciSpaCy imports (graceful fallback if unavailable)
try:
    import spacy  # type: ignore
    import scispacy  # type: ignore  # noqa: F401
    from scispacy.linking import EntityLinker  # type: ignore
    _SCISPACY_AVAILABLE = True
except Exception:
    _SCISPACY_AVAILABLE = False


_NLP = None
_LINKER = None


def _try_load_scispacy_umls() -> Tuple[object, object]:
    """Best-effort load SciSpaCy small model with UMLS linker.

    Returns (nlp, linker) or (None, None) on failure.
    """
    global _NLP, _LINKER
    if not _SCISPACY_AVAILABLE:
        return None, None

    if _NLP is not None and _LINKER is not None:
        return _NLP, _LINKER

    model_name = os.getenv("SCISPACY_MODEL", "en_core_sci_sm")
    try:
        _NLP = spacy.load(model_name)
    except Exception:
        # Model not installed
        return None, None

    try:
        _NLP.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
        _LINKER = _NLP.get_pipe("scispacy_linker")
    except Exception:
        return None, None

    return _NLP, _LINKER


def _normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_symptom_terms_with_umls(note: str, max_terms: int = 50) -> List[str]:
    """Extract symptom-like terms using SciSpaCy + UMLS.

    - Returns a list of normalized preferred names from UMLS linker.
    - Falls back to empty list if SciSpaCy is not available.
    """
    nlp, linker = _try_load_scispacy_umls()
    if nlp is None or linker is None:
        return []

    doc = nlp(note)
    terms: List[str] = []
    seen = set()
    for ent in doc.ents:
        if not getattr(ent._, "kb_ents", None):
            continue
        # take top candidate
        cui, score = ent._.kb_ents[0]
        if score < 0.5:
            continue
        kb_ent = linker.kb.cui_to_entity.get(cui)
        if not kb_ent:
            continue
        name = _normalize_phrase(kb_ent.canonical_name)
        if name and name not in seen:
            seen.add(name)
            terms.append(name)
            if len(terms) >= max_terms:
                break
    return terms


def compute_overlap_score(terms: List[str], title: str, abstract: str) -> Tuple[int, List[str]]:
    """Count how many extracted terms appear in title+abstract (case-insensitive).

    Returns (overlap_count, matched_terms).
    """
    doc_text = f"{title}\n{abstract}".lower()
    matched: List[str] = []
    count = 0
    for term in terms:
        if not term:
            continue
        t = term.lower()
        # simple substring match; could be improved with token-boundary checks
        if t in doc_text:
            matched.append(term)
            count += 1
    return count, matched


def rerank_with_overlap(docs: List[Dict], terms: List[str], threshold: int = 1, top_k: int = 10) -> List[Dict]:
    """Add overlap_count to docs and rerank by (overlap_count, semantic score)."""
    enriched: List[Dict] = []
    for d in docs:
        title = d.get("title", "")
        abstract = d.get("abstract", "")
        overlap_count, matched = compute_overlap_score(terms, title, abstract)
        d = dict(d)
        d["overlap_count"] = int(overlap_count)
        d["matched_terms"] = matched
        enriched.append(d)

    # filter by threshold (soft fallback if too few)
    filtered = [d for d in enriched if d["overlap_count"] >= threshold]
    if len(filtered) < max(3, top_k // 2):
        filtered = enriched

    # sort primarily by overlap_count, then by semantic similarity score if present
    def _score_key(d):
        sim = d.get("score") or d.get("semantic_score") or 0.0
        return (d.get("overlap_count", 0), float(sim))

    reranked = sorted(filtered, key=_score_key, reverse=True)
    return reranked[:top_k]


def build_literature_block(docs: List[Dict], include_overlap: bool = True) -> str:
    chunks = []
    for d in docs:
        base = f"PMID {d['pmid']}: {d.get('title','')}\n{d.get('abstract','')}"
        if include_overlap:
            base += f"\n[overlap_count={d.get('overlap_count',0)} matched_terms={', '.join(d.get('matched_terms', []))}]"
        chunks.append(base)
    return "\n\n".join(chunks)


# -----------------------------
# Candidate disease extraction
# -----------------------------

_DISEASE_HINTS = (
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ syndrome",
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ disease",
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ deficiency",
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ anemia",
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ dystrophy",
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ encephalopathy",
    r"[A-Z][A-Za-z0-9'\-\(\)\s]+ ataxia",
    r"spinocerebellar ataxia[A-Za-z0-9\-\s]*",
    r"SCA\d+",
    r"[A-Z][A-Za-z\-]+-[A-Z][A-Za-z\-]+ syndrome",  # Hyphenated eponyms
    r"22q11\.2[A-Za-z0-9\s]*",  # Chromosomal syndromes
    r"deletion syndrome",
    r"neurodevelopmental disorder[A-Za-z0-9\s,]*",
)

# Generic terms to exclude from candidate extraction
_GENERIC_DISEASE_BLACKLIST = {
    "intellectual disability",
    "developmental delay",
    "congenital abnormality",
    "mental retardation",
    "growth retardation",
    "failure to thrive",
    "global developmental delay",
    "motor delay",
    "speech delay",
    "language delay",
    "cognitive impairment",
    "learning disability",
    "behavioral abnormality",
    "autistic behavior",
    "autism",
    "seizures",
    "epilepsy",
    "hypotonia",
    "hypertonia",
    "spasticity",
    "ataxia",
    "dysarthria",
    "dysphagia",
    "deferred",
    "dwarfism",
    "short stature",
    "microcephaly",
    "macrocephaly",
    "vaccination",
    "vaccination hesitancy",
    "enzyme deficiency",
    "metabolic disorder",
    "genetic disorder",
    "chromosomal abnormality",
    "birth defect",
    "malformation",
    "anomaly",
    "syndrome",
    "disease",
    "disorder",
    "deficiency",
    "abnormality",
    "dysfunction",
    "impairment",
    "disability",
    "retardation",
    "delay"
}

def _is_too_generic(name: str) -> bool:
    """Check if a disease name is too generic to be useful."""
    normalized = _normalize_phrase(name)
    if not normalized:
        return True
    
    # Remove plural 's' for comparison
    normalized_singular = normalized.rstrip('s') if normalized.endswith('s') else normalized
    
    # Check exact match (with and without plural)
    if normalized in _GENERIC_DISEASE_BLACKLIST or normalized_singular in _GENERIC_DISEASE_BLACKLIST:
        return True
    
    # Special cases for terms that should always be filtered
    if any(term in normalized for term in ["deferred", "hesitancy", "unspecified", "other", "nos", "unknown"]):
        return True
    
    # Check if it's just a generic term with a number
    if any(normalized == f"{term} {i}" or normalized_singular == f"{term} {i}" 
           for term in _GENERIC_DISEASE_BLACKLIST for i in range(1, 100)):
        return True
    
    # For multi-word terms, check if it's ONLY generic terms
    words = normalized.split()
    
    # Filter out obvious sentence fragments or paper titles
    if any(word in normalized for word in ["analysis", "clinical", "features", "investigations", 
                                             "revealed", "followed", "presentation", "spectrum",
                                             "suggests", "variants", "associated"]):
        return True
    
    # Allow specific diseases that contain generic words (e.g., "Rett syndrome")
    # These usually have specific identifiers: numbers, eponyms (capitalized in original)
    if len(words) >= 8:
        # Too long - likely a sentence fragment, not a disease name
        return True
    
    if len(words) <= 2:
        # Check if all words are generic
        all_generic = all(w in _GENERIC_DISEASE_BLACKLIST or w.rstrip('s') in _GENERIC_DISEASE_BLACKLIST 
                          for w in words)
        if all_generic:
            # Exception: allow if it contains numbers or hyphens (likely specific)
            if not any(char.isdigit() for char in normalized) and '-' not in normalized:
                return True
        
    return False


def _try_umls_canonical(name: str) -> str:
    nlp, linker = _try_load_scispacy_umls()
    if nlp is None or linker is None:
        return _normalize_phrase(name)
    try:
        doc = nlp(name)
        for ent in doc.ents:
            if getattr(ent._, "kb_ents", None):
                cui, score = ent._.kb_ents[0]
                if score >= 0.5:
                    kb_ent = linker.kb.cui_to_entity.get(cui)
                    if kb_ent and kb_ent.canonical_name:
                        return _normalize_phrase(kb_ent.canonical_name)
    except Exception:
        pass
    return _normalize_phrase(name)


def canonicalize_disease_name(name: str) -> str:
    """Public helper: return UMLS preferred disease name if available, else normalized input."""
    return _try_umls_canonical(name)


def expand_terms_with_umls_synonyms(terms: List[str], max_synonyms_per: int = 2) -> List[str]:
    """Expand input terms with a few UMLS aliases if available."""
    nlp, linker = _try_load_scispacy_umls()
    if nlp is None or linker is None:
        return terms
    expanded: List[str] = []
    seen = set()
    for t in terms:
        if not isinstance(t, str) or not t.strip():
            continue
        base = _normalize_phrase(t)
        if base and base not in seen:
            expanded.append(base)
            seen.add(base)
        try:
            doc = nlp(t)
            for ent in doc.ents:
                if not getattr(ent._, "kb_ents", None):
                    continue
                cui, score = ent._.kb_ents[0]
                if score < 0.5:
                    continue
                kb_ent = linker.kb.cui_to_entity.get(cui)
                if not kb_ent:
                    continue
                for alias in (kb_ent.aliases or [])[:max_synonyms_per]:
                    alias_n = _normalize_phrase(alias)
                    if alias_n and alias_n not in seen:
                        expanded.append(alias_n)
                        seen.add(alias_n)
        except Exception:
            continue
    return expanded


def extract_gene_like_tokens(note: str) -> Set[str]:
    """Very rough gene token detector: uppercase tokens with length 2-10 and digits/dashes allowed."""
    if not isinstance(note, str):
        return set()
    candidates = re.findall(r"\b[A-Z0-9-]{2,10}\b", note)
    common = {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "HAVE", "HAS"}
    genes = set([c for c in candidates if c.isupper() and c not in common])
    return genes


def _normalize_for_match(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[`“”'\"\\]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def score_candidates(
    candidates: List[str],
    docs: List[Dict],
    note: str = "",
    note_hpo_terms: Optional[List[str]] = None,
) -> List[Dict]:
    """Score disease candidates by evidence across retrieved docs.

    Evidence features per doc (min-max normalized when present):
    - semantic sim: d.get('score')
    - cross-encoder score: d.get('ce_score')
    - IDF-weighted term overlap: d.get('overlap_weighted')

    Candidate score = sum over docs containing candidate of
        0.5*sim_n + 0.3*ce_n + 0.2*overlap_n
      + 0.1 * doc_presence_count

    Returns list of {canonical, score, support_pmids, support_titles} sorted desc.
    """
    if not candidates or not docs:
        return []

    # Prepare min-max for features
    sims = [float(d.get("score", 0.0)) for d in docs]
    ces = [float(d.get("ce_score", 0.0)) for d in docs]
    ovs = [float(d.get("overlap_weighted", 0.0)) for d in docs]

    def mm_norm(xs: List[float]) -> Tuple[float, float]:
        x_min = float(min(xs)) if xs else 0.0
        x_max = float(max(xs)) if xs else 1.0
        if x_max - x_min < 1e-9:
            x_max = x_min + 1.0
        return x_min, x_max

    s_min, s_max = mm_norm(sims)
    c_min, c_max = mm_norm(ces)
    o_min, o_max = mm_norm(ovs)

    def norm(x: float, a: float, b: float) -> float:
        return (x - a) / (b - a) if b > a else 0.0

    # Normalize docs text once
    doc_infos = []
    for d in docs:
        text = f"{d.get('title','')}\n{d.get('abstract','')}"
        doc_infos.append({
            "pmid": str(d.get("pmid", "")),
            "title": d.get("title", ""),
            "norm_text": _normalize_for_match(text),
            "sim_n": norm(float(d.get("score", 0.0)), s_min, s_max),
            "ce_n": norm(float(d.get("ce_score", 0.0)), c_min, c_max),
            "ov_n": norm(float(d.get("overlap_weighted", 0.0)), o_min, o_max),
        })

    # Name-agnostic signals: cross-encode (note, candidate name), phenotype overlap
    ce_cand_scores: Dict[str, float] = {}
    if _CROSS_ENCODER_AVAILABLE and note:
        global _CROSS_ENCODER_MODEL
        if _CROSS_ENCODER_MODEL is None:
            try:
                _CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                _CROSS_ENCODER_MODEL = None
        if _CROSS_ENCODER_MODEL is not None:
            pairs = [(note, c) for c in candidates]
            try:
                scores = _CROSS_ENCODER_MODEL.predict(pairs)
                for c, s in zip(candidates, scores):
                    ce_cand_scores[c] = float(s)
            except Exception:
                pass

    # Phenotype overlap using dataset-provided HPO ids
    _, d2h = _load_dataset_label_map()
    cand_pheno_scores: Dict[str, float] = {}
    note_term_set = set([_normalize_for_match(t) for t in (note_hpo_terms or []) if t])
    for c in candidates:
        mapped = map_to_dataset_label(c)
        if mapped and mapped in d2h:
            # Use label string tokens as proxy + HPO ids count
            lab_tokens = set(_normalize_for_match(mapped).split())
            token_overlap = len(lab_tokens & note_term_set) / float(len(lab_tokens | note_term_set) or 1)
            hpo_count = len(d2h.get(mapped, set()))
            cand_pheno_scores[c] = 0.5 * token_overlap + 0.5 * (min(50, hpo_count) / 50.0)
        else:
            cand_pheno_scores[c] = 0.0

    # Aggregate by canonical candidate
    bucket: Dict[str, Dict] = {}
    for cand in candidates:
        canon = canonicalize_disease_name(cand)
        canon_n = _normalize_for_match(canon)
        if not canon_n:
            continue
        if canon not in bucket:
            bucket[canon] = {
                "canonical": canon,
                "score": 0.0,
                "support": [],  # tuples (score, pmid, title)
                "presence": 0,
            }
        # Score presence across docs
        for info in doc_infos:
            # Doc-level support does not require literal name mention; use doc scores directly
            doc_score = 0.5 * info["sim_n"] + 0.3 * info["ce_n"] + 0.2 * info["ov_n"]
            bucket[canon]["score"] += doc_score
            bucket[canon]["presence"] += 1
            bucket[canon]["support"].append((doc_score, info["pmid"], info["title"]))

        # Add name-agnostic candidate signals
        bucket[canon]["score"] += 0.5 * float(ce_cand_scores.get(cand, 0.0))
        bucket[canon]["score"] += 0.5 * float(cand_pheno_scores.get(cand, 0.0))

    # Finalize scores and supports
    results: List[Dict] = []
    for canon, data in bucket.items():
        final_score = float(data["score"] + 0.1 * data["presence"])
        support_sorted = sorted(data["support"], key=lambda x: x[0], reverse=True)
        pmids = [pmid for _, pmid, _ in support_sorted[:5] if pmid]
        titles = [title for _, _, title in support_sorted[:5] if title]
        results.append({
            "canonical": canon,
            "score": final_score,
            "support_pmids": pmids,
            "support_titles": titles,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def extract_disease_candidates_from_docs(
    docs: List[Dict],
    max_candidates: int = 30,
    max_synonyms_per: int = 3,
    use_umls: bool = True,
) -> List[str]:
    """Mine disease candidates from titles/abstracts and canonicalize.

    Preference order:
    1) UMLS linking to disease-like semantic types, with aliases as synonyms
    2) Regex fallback patterns (e.g., "X syndrome")
    Returns unique candidates (canonical + few synonyms) sorted by frequency.
    """
    counts: Dict[str, int] = {}

    # Try UMLS-driven extraction
    if use_umls and _SCISPACY_AVAILABLE:
        nlp, linker = _try_load_scispacy_umls()
        if nlp is not None and linker is not None:
            allowed_types = {"T047", "T048", "T046", "T191", "T019"}
            seen_names: Dict[str, int] = {}
            for d in docs:
                text = f"{d.get('title','')}\n{d.get('abstract','')}"
                try:
                    doc = nlp(text)
                except Exception:
                    continue
                for ent in doc.ents:
                    if not getattr(ent._, "kb_ents", None):
                        continue
                    cui, score = ent._.kb_ents[0]
                    if score < 0.7:  # Increased threshold to reduce noise
                        continue
                    kb_ent = linker.kb.cui_to_entity.get(cui)
                    if not kb_ent:
                        continue
                    if not (set(kb_ent.types or []) & allowed_types):
                        continue
                    # Canonical + a few aliases
                    canonical = _normalize_phrase(kb_ent.canonical_name)
                    if canonical and not _is_too_generic(canonical):
                        counts[canonical] = counts.get(canonical, 0) + 1
                    for alias in (kb_ent.aliases or [])[:max_synonyms_per]:
                        alias_n = _normalize_phrase(alias)
                        if alias_n and alias_n != canonical and not _is_too_generic(alias_n):
                            counts[alias_n] = counts.get(alias_n, 0) + 1

            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            result = [name for name, _ in ranked][:max_candidates]
            if len(result) < 3:
                print(f"⚠️ Only {len(result)} UMLS candidates found after filtering (had {len(counts)} before filtering)")
                # Fall through to regex fallback
            else:
                return result

    # Regex fallback
    patterns = [re.compile(pat) for pat in _DISEASE_HINTS]
    for d in docs:
        text = f"{d.get('title','')}\n{d.get('abstract','')}"
        for pat in patterns:
            for m in pat.findall(text):
                cand = _try_umls_canonical(m)
                if not cand or _is_too_generic(cand):
                    continue
                counts[cand] = counts.get(cand, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked][:max_candidates]


def extract_disease_candidates_from_note(note: str, max_candidates: int = 20) -> List[str]:
    """Extract disease-like candidates directly from the note via UMLS and regex."""
    counts: Dict[str, int] = {}
    # UMLS
    nlp, linker = _try_load_scispacy_umls()
    if nlp is not None and linker is not None:
        try:
            doc = nlp(note)
            allowed_types = {"T047", "T048", "T046", "T191", "T019"}
            for ent in doc.ents:
                if not getattr(ent._, "kb_ents", None):
                    continue
                cui, score = ent._.kb_ents[0]
                if score < 0.5:
                    continue
                kb_ent = linker.kb.cui_to_entity.get(cui)
                if not kb_ent:
                    continue
                if not (set(kb_ent.types or []) & allowed_types):
                    continue
                canonical = _normalize_phrase(kb_ent.canonical_name)
                if canonical and not _is_too_generic(canonical):
                    counts[canonical] = counts.get(canonical, 0) + 1
        except Exception:
            pass

    # Regex fallback
    patterns = [re.compile(p) for p in _DISEASE_HINTS]
    for pat in patterns:
        for m in pat.findall(note or ""):
            cand = _try_umls_canonical(m)
            if not cand or _is_too_generic(cand):
                continue
            counts[cand] = counts.get(cand, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked][:max_candidates]


def augment_with_idf_weighted_overlap(docs: List[Dict], terms: List[str]) -> List[Dict]:
    """Compute IDF-weighted overlap across the current doc set and annotate docs.

    IDF is approximated from document frequency within this candidate set.
    """
    if not docs:
        return docs
    term_set = [t.lower() for t in terms if t]
    if not term_set:
        for d in docs:
            d["overlap_weighted"] = 0.0
        return docs

    # Document frequency per term (within retrieved set)
    dfreq: Dict[str, int] = {t: 0 for t in term_set}
    for d in docs:
        text = f"{d.get('title','')}\n{d.get('abstract','')}".lower()
        present = set()
        for t in term_set:
            if t in text:
                present.add(t)
        for t in present:
            dfreq[t] += 1

    N = max(1, len(docs))
    idf: Dict[str, float] = {t: np.log((N + 1) / (df + 1)) + 1.0 for t, df in dfreq.items()}

    for d in docs:
        text = f"{d.get('title','')}\n{d.get('abstract','')}".lower()
        score = 0.0
        for t in term_set:
            if t in text:
                score += idf.get(t, 1.0)
        d["overlap_weighted"] = float(score)
    return docs


_CROSS_ENCODER_MODEL: Optional[object] = None

# -----------------------------
# Dataset-derived label map (for canonicalization and HPO phenotype sets)
# -----------------------------
_ALL_LABELS: Optional[List[str]] = None
_DISEASE_TO_HPOIDS: Optional[Dict[str, Set[str]]] = None


def _normalize_label(text: str) -> str:
    return _normalize_for_match(text)


def _load_dataset_label_map() -> Tuple[List[str], Dict[str, Set[str]]]:
    global _ALL_LABELS, _DISEASE_TO_HPOIDS
    if _ALL_LABELS is not None and _DISEASE_TO_HPOIDS is not None:
        return _ALL_LABELS, _DISEASE_TO_HPOIDS
    try:
        df = pd.read_csv("Datasets/subset_all_mapped.csv")
    except Exception:
        _ALL_LABELS = []
        _DISEASE_TO_HPOIDS = {}
        return _ALL_LABELS, _DISEASE_TO_HPOIDS

    labels: List[str] = []
    d2h: Dict[str, Set[str]] = {}
    for row in df.itertuples(index=False):
        disease = getattr(row, "disease", None)
        if not isinstance(disease, str) or not disease.strip():
            continue
        labels.append(disease)
        # parse hpo ids list if present
        ids_raw = getattr(row, "matched_hpo_ids", None)
        hpo_ids: Set[str] = set()
        if isinstance(ids_raw, str) and ids_raw.strip():
            try:
                import ast as _ast
                ids = _ast.literal_eval(ids_raw)
                for x in ids:
                    if isinstance(x, str):
                        hpo_ids.add(x.strip())
                    elif isinstance(x, list) and x:
                        hpo_ids.add(str(x[0]).strip())
            except Exception:
                pass
        d2h.setdefault(disease, set()).update(hpo_ids)

    _ALL_LABELS = sorted(set(labels))
    _DISEASE_TO_HPOIDS = d2h
    return _ALL_LABELS, _DISEASE_TO_HPOIDS


def map_to_dataset_label(name: str) -> Optional[str]:
    """Map a canonical disease name to the closest dataset label using token Jaccard."""
    labels, _ = _load_dataset_label_map()
    if not labels:
        return None
    target = _normalize_label(name)
    if not target:
        return None
    def toks(s: str) -> Set[str]:
        return set([t for t in _normalize_label(s).split() if t])
    tset = toks(target)
    best = None
    best_sim = 0.0
    for lab in labels:
        lset = toks(lab)
        if not (tset or lset):
            continue
        sim = len(tset & lset) / len(tset | lset)
        if sim > best_sim:
            best_sim = sim
            best = lab
    # require a minimal similarity to avoid wild mappings
    return best if (best and best_sim >= 0.3) else None


def rerank_with_cross_encoder(query: str, docs: List[Dict], top_k: int = 10, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List[Dict]:
    """Rerank docs using a cross-encoder on (query, title+abstract)."""
    if not _CROSS_ENCODER_AVAILABLE:
        return docs[:top_k]
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        try:
            _CROSS_ENCODER_MODEL = CrossEncoder(model_name)
        except Exception:
            return docs[:top_k]
    pairs = []
    for d in docs:
        text = f"{d.get('title','')}\n{d.get('abstract','')}"
        pairs.append((query, text))
    try:
        scores = _CROSS_ENCODER_MODEL.predict(pairs)
    except Exception:
        return docs[:top_k]
    for d, s in zip(docs, scores):
        d["ce_score"] = float(s)
    reranked = sorted(docs, key=lambda x: x.get("ce_score", 0.0), reverse=True)
    return reranked[:top_k]


