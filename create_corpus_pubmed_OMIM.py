import os
import pandas as pd
import json
import time
import requests
import re
from tqdm import tqdm
from Bio import Entrez
from rapidfuzz import process
from bs4 import BeautifulSoup
from lxml import etree

# -------------------- CONFIG --------------------
from dotenv import load_dotenv
load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL", "your-email@example.com")
OMIM_API_KEY = os.getenv("OMIM_API_KEY")
DATASET_FILE = "subset.csv"
OUTPUT_FILE = "subset_corpus.jsonl"
MAX_PUBMED_RESULTS = 50
SNAPSHOT_INTERVAL = 200
PROGRESS_INTERVAL = 50

# -------------------- LOAD DATA --------------------
df = pd.read_csv(DATASET_FILE)
search_terms = set(df['disease'].dropna().unique())
df['clean_omim'] = df['omim_id'].str.replace("OMIM:", "", regex=False)
omim_ids = df['clean_omim'].dropna().astype(str).unique()
pmids = df['pmid'].dropna().astype(str).str.extract(r'(\d+)', expand=False).dropna().unique().tolist()

# -------------------- UTILITIES --------------------
def fetch_pubmed_ids(term, max_results=MAX_PUBMED_RESULTS):
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
        record = Entrez.read(handle)
        return record["IdList"]
    except Exception as e:
        print(f"❌ Error searching PubMed for term: {term}\n{e}")
        return []

def fetch_pubmed_details(id_list):
    results = []
    if not id_list:
        return results
    ids = ",".join(id_list)
    try:
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        for article in records['PubmedArticle']:
            try:
                pmid = str(article['MedlineCitation']['PMID'])
                title = article['MedlineCitation']['Article']['ArticleTitle']
                abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                results.append({"pmid": pmid, "title": title, "abstract": abstract})
            except:
                continue
        return results
    except Exception as e:
        print(f"❌ Error fetching details for PMIDs: {id_list[:5]}...\n{e}")
        return []

def fetch_omim_pmids(omim_id):
    try:
        url = f"https://api.omim.org/api/entry?mimNumber={omim_id}&include=referenceList&format=json&apiKey={OMIM_API_KEY}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        pmids = []
        ref_list = data.get('omim', {}).get('entryList', [])[0].get('entry', {}).get('referenceList', [])
        for ref in ref_list:
            pmid = ref.get('reference', {}).get('pubmedID')
            if pmid:
                pmids.append(str(pmid))
        return pmids
    except:
        return []

def clean_synopsis_text(text):
    return re.sub(r"\{.*?\}", "", text).strip()

def fetch_omim_summary(omim_id):
    try:
        url = f"https://api.omim.org/api/entry?mimNumber={omim_id}&include=textSection,clinicalSynopsis&format=json&apiKey={OMIM_API_KEY}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        entry = data.get("omim", {}).get("entryList", [])[0].get("entry", {})
        title = entry.get("titles", {}).get("preferredTitle", "")
        pmid = f"OMIM:{omim_id}"

        abstract_parts = []

        sections = entry.get("textSectionList", [])
        for section in sections:
            sec = section.get("textSection", {})
            text = sec.get("text")
            if text:
                abstract_parts.append(f"{sec.get('heading', '')}: {text}")

        if not abstract_parts:
            cs = entry.get("clinicalSynopsis", {})
            for k, v in cs.items():
                if isinstance(v, str) and v.strip() and not k.startswith("includes"):
                    key = k.replace('_', ' ').capitalize()
                    abstract_parts.append(f"{key}: {v.strip()}")

        abstract = "\n\n".join(abstract_parts).strip()
        abstract = clean_synopsis_text(abstract)

        if not abstract:
            print(f"⚠️ OMIM {omim_id} has no usable text.")
            return None

        return {
            "pmid": pmid,
            "title": f"OMIM Entry for {title}",
            "abstract": abstract,
            "source": "OMIM"
        }
    except Exception as e:
        print(f"❌ OMIM summary fetch failed for {omim_id}: {e}")
        return None

# -------------------- BUILD CORPUS --------------------
corpus = []
fetched_pmids = set()

# 1. PubMed keyword search
print("🔍 Searching PubMed by keywords...")
for term in tqdm(search_terms):
    ids = fetch_pubmed_ids(term)
    time.sleep(0.4)
    ids = [i for i in ids if i not in fetched_pmids]
    entries = fetch_pubmed_details(ids)
    corpus.extend(entries)
    fetched_pmids.update(ids)

# 2. Direct PMIDs
print("📄 Fetching abstracts by PMIDs...")
for i in tqdm(range(0, len(pmids), 100)):
    chunk = [pid for pid in pmids[i:i+100] if pid not in fetched_pmids]
    entries = fetch_pubmed_details(chunk)
    corpus.extend(entries)
    fetched_pmids.update(chunk)

# 3. OMIM-linked abstracts
print("🧬 Fetching OMIM-linked abstracts...")
for omim_id in tqdm(omim_ids):
    omim_pmids = fetch_omim_pmids(omim_id)
    new_pmids = [pid for pid in omim_pmids if pid not in fetched_pmids]
    entries = fetch_pubmed_details(new_pmids)
    corpus.extend(entries)
    fetched_pmids.update(new_pmids)

# 4. OMIM summaries
print("📚 Fetching OMIM summaries...")
for omim_id in tqdm(omim_ids):
    summary = fetch_omim_summary(omim_id)
    if summary:
        corpus.append(summary)

# -------------------- SAVE CORPUS --------------------
print(f"💾 Saving {len(corpus)} entries to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in corpus:
        json.dump(entry, f)
        f.write("\n")

print("✅ Corpus creation complete.")