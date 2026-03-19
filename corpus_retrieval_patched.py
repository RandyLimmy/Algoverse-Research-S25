import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import ast
import re
import math
from collections import Counter, defaultdict


def is_valid_doc(doc):
    bad_keywords = ["erratum", "correction", "corrigendum", "notice"]
    abstract = doc.get("abstract", "")
    title = doc.get("title", "")
    return (
        abstract and len(abstract.split()) > 50 and
        all(bad not in title.lower() for bad in bad_keywords)
    )

# -------------------
# ✅ Device Setup
# -------------------
device = torch.device("cpu")  # Use CPU to avoid GPU-related crashes on macOS

# -------------------
# ✅ Load Models
# -------------------
# -------------------
# ✅ Load Models Safely
# -------------------
try:
    pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    pubmedbert_model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        trust_remote_code=True,
        use_safetensors=True
    ).to(device)
    pubmedbert_model.eval()
except Exception as e:
    print(f"❌ Failed to load PubMedBERT: {e}")
    exit(1)

try:
    sapbert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    sapbert_model = AutoModel.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        trust_remote_code=True,
        use_safetensors=True
    ).to(device)
    sapbert_model.eval()
except Exception as e:
    print(f"❌ Failed to load SapBERT: {e}")
    exit(1)

# -------------------
# ✅ Load Index + Corpus
# -------------------
pubmedbert_index = faiss.read_index("Datasets/indexes/pubmedbert.index")
sapbert_index = faiss.read_index("Datasets/indexes/sapbert.index")

with open("Datasets/retrieval_corpus/pmid_mapping.json", "r") as f:
    pmid_list = json.load(f)

with open("Datasets/retrieval_corpus/corpus_subset.jsonl", "r") as f:
    corpus = [json.loads(line) for line in f]

# Fast lookup for docs by PMID and index mapping
pmid_to_idx = {str(pmid): i for i, pmid in enumerate(pmid_list)}
pmid_to_doc = {str(doc.get("pmid")): doc for doc in corpus}

# -------------------
# ✅ Lightweight BM25 Index (in-memory)
# -------------------
_BM25_READY = False
_BM25_POSTINGS: dict[str, list[tuple[int, int]]] = {}
_BM25_DF: dict[str, int] = {}
_BM25_DOC_LEN: list[int] = []
_BM25_AVGDL: float = 0.0


def _bm25_tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def _ensure_bm25_index():
    global _BM25_READY, _BM25_POSTINGS, _BM25_DF, _BM25_DOC_LEN, _BM25_AVGDL
    if _BM25_READY:
        return
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    df: dict[str, int] = defaultdict(int)
    doc_len: list[int] = []

    for i, doc in enumerate(corpus):
        text = f"{doc.get('title','')}\n{doc.get('abstract','')}"
        toks = _bm25_tokenize(text)
        doc_len.append(len(toks))
        counts = Counter(toks)
        for term, tf in counts.items():
            postings[term].append((i, int(tf)))
        for term in counts.keys():
            df[term] += 1

    _BM25_POSTINGS = postings
    _BM25_DF = df
    _BM25_DOC_LEN = doc_len
    _BM25_AVGDL = (sum(doc_len) / max(1, len(doc_len))) if doc_len else 0.0
    _BM25_READY = True


def retrieve_bm25(query_text: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
    """Retrieve docs using BM25 over title+abstract.

    Returns list of dicts with fields: pmid, bm25, title, abstract
    """
    _ensure_bm25_index()
    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        return []
    toks = _bm25_tokenize(query_text)
    if not toks:
        return []

    scores: dict[int, float] = defaultdict(float)
    N = len(_BM25_DOC_LEN)
    avgdl = _BM25_AVGDL
    # Precompute IDF for query terms
    for term in set(toks):
        df = _BM25_DF.get(term, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for doc_id, tf in _BM25_POSTINGS.get(term, []):
            dl = _BM25_DOC_LEN[doc_id]
            denom = tf + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
            scores[doc_id] += idf * (tf * (k1 + 1) / max(1e-9, denom))

    # Rank by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, sc in ranked[: max(top_k, 0)]:
        pmid = str(pmid_list[doc_id])
        doc = pmid_to_doc.get(pmid)
        if not doc:
            continue
        if not is_valid_doc(doc):
            continue
        abstract = doc.get("abstract", "").replace("\n", " ").strip() or "(No abstract available)"
        dd = dict(doc)
        dd["abstract"] = abstract
        results.append({"pmid": pmid, "bm25": float(sc), **dd})
    return results

# -------------------
# ✅ Embedding Function
# -------------------
def embed_text(text, model, tokenizer, max_length=384):
    if not text or not isinstance(text, str) or text.strip() == "":
        return np.zeros((1, 768), dtype=np.float32)

    try:
        text = text.replace("\n", " ").replace("  ", " ").strip()

        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state.squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)

            valid_hidden = last_hidden_state[attention_mask.bool()]
            if valid_hidden.size(0) == 0:
                return np.zeros((1, 768), dtype=np.float32)

            vec = valid_hidden.mean(dim=0).cpu().numpy().reshape(1, -1)
            # L2-normalize and cast to float32 for FAISS IP search
            denom = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
            vec = (vec / denom).astype(np.float32)
            return vec

    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return np.zeros((1, 768), dtype=np.float32)


# -------------------
# ✅ Retrieval Functions
# -------------------
def retrieve_pubmedbert(case_text, top_k=10):
    case_vec = embed_text(case_text, pubmedbert_model, pubmedbert_tokenizer)
    scores, indices = pubmedbert_index.search(case_vec, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        pmid = pmid_list[idx]
        try:
            doc = next(doc for doc in corpus if doc["pmid"] == pmid)
        except StopIteration:
            print(f"⚠️ Skipping (not found): {pmid}")
            continue

        abstract = doc.get("abstract", "").replace("\n", " ").strip() or "(No abstract available)"
        doc["abstract"] = abstract

        results.append({"pmid": pmid, "score": float(score), **doc})
    return results

def retrieve_hybrid(case_text, hpo_terms, top_k=10, sem_weight: float = 0.6, bm25_weight: float = 0.2, ont_weight: float | None = None, fusion_mode: str = "normalized"):
    """Hybrid retrieval over semantic, ontology, and lexical (BM25).

    We min-max normalize each channel per query and fuse with weights.
    ont_weight defaults to (1 - sem_weight - bm25_weight) if not provided.
    """
    if ont_weight is None:
        ont_weight = max(0.0, 1.0 - float(sem_weight) - float(bm25_weight))

    sem_vec = embed_text(case_text, pubmedbert_model, pubmedbert_tokenizer)
    ont_vec = embed_text(" ".join(hpo_terms), sapbert_model, sapbert_tokenizer)

    sem_scores, sem_indices = pubmedbert_index.search(sem_vec, max(top_k, 0))
    ont_scores, ont_indices = sapbert_index.search(ont_vec, max(top_k, 0))

    # Legacy fusion (raw FAISS scores, no BM25)
    if fusion_mode == "legacy":
        combined = {}
        for idx, score in zip(sem_indices[0], sem_scores[0]):
            combined[int(idx)] = {"sem_raw": float(score), "ont_raw": 0.0}
        for idx, score in zip(ont_indices[0], ont_scores[0]):
            row = combined.get(int(idx), {"sem_raw": 0.0, "ont_raw": 0.0})
            row["ont_raw"] = float(score)
            combined[int(idx)] = row
        fused = []
        ont_w = 1.0 - float(sem_weight)
        for idx, ch in combined.items():
            s = float(sem_weight) * ch.get("sem_raw", 0.0) + float(ont_w) * ch.get("ont_raw", 0.0)
            fused.append((idx, s, ch))
        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score, ch in fused[: max(top_k, 0)]:
            pmid = pmid_list[idx]
            doc = pmid_to_doc.get(str(pmid))
            if not doc:
                continue
            if not is_valid_doc(doc):
                continue
            abstract = doc.get("abstract", "").replace("\n", " ").strip() or "(No abstract available)"
            dd = dict(doc)
            dd["abstract"] = abstract
            results.append({
                "pmid": pmid,
                "score": float(score),
                "semantic_score": float(ch.get("sem_raw", 0.0)),
                "ontology_score": float(ch.get("ont_raw", 0.0)),
                **dd
            })
        return results

    # Normalized fusion with BM25
    # Normalize per channel
    def _norm(arr: np.ndarray) -> np.ndarray:
        if arr is None or len(arr[0]) == 0:
            return arr
        v = arr[0]
        vmin = float(np.min(v)) if len(v) else 0.0
        vmax = float(np.max(v)) if len(v) else 1.0
        if vmax - vmin < 1e-9:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    sem_n = _norm(sem_scores)
    ont_n = _norm(ont_scores)

    # BM25 over combined text (note + HPO terms)
    bm25_query = (case_text or "") + "\n" + (" ".join(hpo_terms) if hpo_terms else "")
    bm25_hits = retrieve_bm25(bm25_query, top_k=max(top_k, 0))
    bm25_pmids = [str(d.get("pmid")) for d in bm25_hits]
    bm25_vals = [float(d.get("bm25", 0.0)) for d in bm25_hits]
    if bm25_vals:
        bmin, bmax = min(bm25_vals), max(bm25_vals)
        if bmax - bmin < 1e-9:
            bm25_norm_vals = [0.0 for _ in bm25_vals]
        else:
            bm25_norm_vals = [(v - bmin) / (bmax - bmin) for v in bm25_vals]
    else:
        bm25_norm_vals = []

    # Build combined map keyed by FAISS index when available, else PMID only
    combined: dict[int, dict] = {}
    # Semantic channel
    for pos, (idx, _) in enumerate(zip(sem_indices[0], sem_scores[0])):
        combined[int(idx)] = {"sem_n": float(sem_n[pos]) if len(sem_n) else 0.0, "ont_n": 0.0, "bm25_n": 0.0}
    # Ontology channel
    for pos, (idx, _) in enumerate(zip(ont_indices[0], ont_scores[0])):
        row = combined.get(int(idx), {"sem_n": 0.0, "ont_n": 0.0, "bm25_n": 0.0})
        row["ont_n"] = float(ont_n[pos]) if len(ont_n) else row["ont_n"]
        combined[int(idx)] = row
    # BM25 channel mapped to indices via PMID
    for pmid, bn in zip(bm25_pmids, bm25_norm_vals):
        i = pmid_to_idx.get(str(pmid))
        if i is None:
            continue
        row = combined.get(int(i), {"sem_n": 0.0, "ont_n": 0.0, "bm25_n": 0.0})
        row["bm25_n"] = float(bn)
        combined[int(i)] = row

    # Fuse scores
    fused = []
    for idx, ch in combined.items():
        s = float(sem_weight) * ch.get("sem_n", 0.0) + float(bm25_weight) * ch.get("bm25_n", 0.0) + float(ont_weight) * ch.get("ont_n", 0.0)
        fused.append((idx, s, ch))
    fused.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score, ch in fused[: max(top_k, 0)]:
        pmid = pmid_list[idx]
        doc = pmid_to_doc.get(str(pmid))
        if not doc:
            continue
        if not is_valid_doc(doc):
            continue
        abstract = doc.get("abstract", "").replace("\n", " ").strip() or "(No abstract available)"
        dd = dict(doc)
        dd["abstract"] = abstract
        results.append({
            "pmid": pmid,
            "score": float(score),
            "semantic_score": float(ch.get("sem_n", 0.0)),
            "ontology_score": float(ch.get("ont_n", 0.0)),
            "bm25_score": float(ch.get("bm25_n", 0.0)),
            **dd
        })
    return results

def main():
    import pandas as pd
    import ast

    # Load your real case dataset
    df = pd.read_csv("Datasets/subset_all_mapped.csv")

    # Only process the first 5 rows
    for idx, row in df.head(2).iterrows():
        print(f"\n======= Case {idx} =======")
        case_text = row["clinical_note"]

        # Parse HPO terms
        try:
            matched_pairs = ast.literal_eval(row["matched_hpo_terms"])
            hpo_terms = [term[0] if isinstance(term, list) else term for term in matched_pairs]
        except Exception as e:
            print(f"⚠️ Skipping row {idx}: error parsing HPO terms → {e}")
            continue

        print("HPO terms for SapBERT:", hpo_terms)

        # Optional: enrich case with extracted symptoms
        try:
            extracted = ast.literal_eval(row["extracted_symptoms"])
            case_text += " Symptoms: " + ", ".join(extracted)
        except Exception:
            print(f"⚠️ No extracted symptoms found or parsing failed for row {idx}")

        # Semantic-only retrieval
        print("🔹 Semantic-only Retrieval (PubMedBERT):")
        top_semantic_raw = retrieve_pubmedbert(case_text, top_k=20)
        top_semantic = [doc for doc in top_semantic_raw if is_valid_doc(doc)][:5]
        for i, doc in enumerate(top_semantic, 1):
            print(f"\n🔸 Result {i}: PMID {doc['pmid']}")
            print(f"Semantic Score: {doc['score']:.2f}")
            print("Title:", doc["title"])
            abstract = doc["abstract"].replace("\n", " ").strip()
            print("Abstract:", abstract[:500] + "...")

        # Hybrid retrieval
        print("\n🔸 Hybrid Retrieval (PubMedBERT + SapBERT):")
        top_hybrid_raw = retrieve_hybrid(case_text, hpo_terms, top_k=20)
        top_hybrid = [doc for doc in top_hybrid_raw if is_valid_doc(doc)][:5]
        for i, doc in enumerate(top_hybrid, 1):
            print(f"\n🔸 Result {i}: PMID {doc['pmid']}")
            print(f"Semantic Score: {doc['semantic_score']:.2f}")
            print(f"Ontology Score: {doc['ontology_score']:.2f}")
            print("Title:", doc["title"])
            abstract = doc["abstract"].replace("\n", " ").strip()
            print("Abstract:", abstract[:500] + "...")

# Entry point
if __name__ == "__main__":
    main()
