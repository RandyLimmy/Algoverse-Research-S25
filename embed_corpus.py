import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ----------- Load Corpus ------------
with open("Datasets/retrieval_corpus/corpus_subset.jsonl", "r") as f:
    corpus = [json.loads(line) for line in f]

# ----------- Load Models ------------
# PubMedBERT
pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
pubmedbert_model = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    use_safetensors=True,
    trust_remote_code=True
)
pubmedbert_model.eval()

# SapBERT
try:
    sapbert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    sapbert_model = AutoModel.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        use_safetensors=True,
        trust_remote_code=True
    )
    sapbert_model.eval()
except Exception as e:
    print(f"❌ SapBERT model loading failed: {e}")
    exit(1)

# ----------- Device Setup ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pubmedbert_model = pubmedbert_model.to(device)
sapbert_model = sapbert_model.to(device)

# ----------- Embedding Function (masked mean + unit norm) ------------
def get_embedding(text, model, tokenizer, max_length: int = 384):
    """Embed text using attention-masked mean pooling and L2 normalization.

    Uses the same pooling/normalization as query-time to ensure parity.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        # Return a safe unit vector (all zeros normalized stays zeros)
        zero_vec = torch.zeros(1, model.config.hidden_size, device=device)
        return F.normalize(zero_vec, p=2, dim=1).squeeze(0).cpu().numpy()

    clean_text = text.replace("\n", " ").strip()
    tokens = tokenizer(
        clean_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)

    last_hidden = output.last_hidden_state  # [1, seq, dim]
    mask = tokens["attention_mask"].unsqueeze(-1).float()  # [1, seq, 1]
    masked_sum = (last_hidden * mask).sum(dim=1)            # [1, dim]
    denom = mask.sum(dim=1).clamp(min=1e-9)                 # [1, 1]
    pooled = masked_sum / denom                             # [1, dim]
    pooled = F.normalize(pooled, p=2, dim=1)                # unit-norm
    return pooled.squeeze(0).cpu().numpy()

# ----------- Initialize FAISS Indexes (cosine via inner product) ------------
pubmedbert_index = faiss.IndexFlatIP(768)
sapbert_index = faiss.IndexFlatIP(768)
pmid_list = []

# ----------- Embed All Abstracts ------------
for entry in tqdm(corpus, desc="🔍 Embedding corpus"):
    pmid = entry["pmid"]
    abstract = entry["abstract"]

    pubmed_vec = get_embedding(abstract, pubmedbert_model, pubmedbert_tokenizer)
    sapbert_vec = get_embedding(abstract, sapbert_model, sapbert_tokenizer)

    pubmedbert_index.add(pubmed_vec.reshape(1, -1))
    sapbert_index.add(sapbert_vec.reshape(1, -1))
    pmid_list.append(pmid)

# ----------- Save Indexes and Metadata ------------
faiss.write_index(pubmedbert_index, "Datasets/indexes/pubmedbert.index")
faiss.write_index(sapbert_index, "Datasets/indexes/sapbert.index")

with open("Datasets/retrieval_corpus/pmid_mapping.json", "w") as f:
    json.dump(pmid_list, f)

print("✅ Embedding complete. FAISS indexes and PMIDs saved!")
