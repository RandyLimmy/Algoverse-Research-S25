import pandas as pd
import ast
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- 📂 Load data ----------
notes_df = pd.read_csv('Datasets/uncleaned_dataset.csv')
hpo_df = pd.read_csv('Datasets/hpo_terms.csv')

# ---------- 🧠 Preprocess HPO terms into searchable lookup ----------
hpo_lookup = {}

for _, row in hpo_df.iterrows():
    if pd.notnull(row['name']):
        hpo_lookup[row['name'].strip().lower()] = row['id']

    if pd.notnull(row['synonyms']):
        try:
            synonyms = ast.literal_eval(row['synonyms'])
            for syn in synonyms:
                hpo_lookup[syn.strip().lower()] = row['id']
        except Exception as e:
            print(f"❌ Failed to parse synonyms for {row['name']}: {e}")

    if pd.notnull(row['definition']):
        hpo_lookup[row['definition'].strip().lower()] = row['id']

lookup_keys = list(hpo_lookup.keys())  # Precompute keys for fuzzy/semantic match

# ---------- 🤖 Extract symptoms from GPT ----------
def extract_symptoms(note):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant. Extract a list of observable patient symptoms or clinical findings "
                        "from the following clinical note. Return the result as a JSON list like this: "
                        "[\"symptom 1\", \"symptom 2\", ...]. Do not include code formatting."
                    )
                },
                {"role": "user", "content": note}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        symptoms = json.loads(content)
        if not isinstance(symptoms, list):
            raise ValueError("Expected a list of symptoms.")
        return symptoms
    except Exception as e:
        print(f"❌ OpenAI error or parsing issue: {e}")
        return []

# ---------- 🔁 GPT fallback for semantic match ----------
def get_top_candidates(symptom, hpo_dict, max_candidates=30):
    from rapidfuzz.fuzz import partial_ratio

    ranked = []
    for term, hpo_id in hpo_dict.items():
        word_overlap = len(set(symptom.lower().split()) & set(term.lower().split()))
        fuzzy_score = partial_ratio(symptom.lower(), term.lower())
        combined_score = word_overlap + (fuzzy_score / 100.0)  # weighted combo
        ranked.append((term, hpo_id, combined_score))

    ranked.sort(key=lambda x: -x[2])
    return [(term, hpo_id) for term, hpo_id, _ in ranked[:max_candidates]]

def gpt_semantic_hpo_match(symptom, hpo_dict, max_candidates=30):
    candidates = get_top_candidates(symptom, hpo_dict, max_candidates)
    options = "\n".join([f"{term} ({hpo_id})" for term, hpo_id in candidates])
    prompt = (
        f"A symptom was extracted: '{symptom}'. From the list below, return the best matching HPO ID, "
        f"or 'None' if there's no match.\n\n"
        f"{options}\n\n"
        f"Respond only with the matching HPO ID (e.g., HP:0000729), or None."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        output = response.choices[0].message.content.strip()
        if output.startswith("HP:"):
            return output
        return None
    except Exception as e:
        print(f"⚠️ GPT fallback error for '{symptom}': {e}")
        return None

# ---------- 🔍 Match extracted symptoms to HPO terms ----------
def match_to_hpo(symptom_list):
    matches = []
    for symptom in symptom_list:
        symptom_clean = symptom.strip().lower()

        # Direct match
        if symptom_clean in hpo_lookup:
            matches.append((symptom, hpo_lookup[symptom_clean]))
            continue

        # Fuzzy match
        best_match = process.extractOne(
            symptom_clean,
            lookup_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85
        )
        if best_match:
            matches.append((symptom, hpo_lookup[best_match[0]]))
            continue

        # GPT semantic fallback
        gpt_match = gpt_semantic_hpo_match(symptom_clean, hpo_lookup)
        if gpt_match:
            matches.append((symptom, gpt_match))
        else:
            print(f"🔍 No match found for: '{symptom}'")
            matches.append((symptom, None))
    return matches

# ---------- 🔄 Process each row ----------
extracted_list = []
hpo_matches_list = []

for i, note in enumerate(notes_df['clinical_note']):
    print(f"📄 Processing row {i+1}/{len(notes_df)}...")
    symptoms = extract_symptoms(note)
    hpo_matches = match_to_hpo(symptoms)

    extracted_list.append(symptoms)
    hpo_matches_list.append(hpo_matches)

    # ✅ Save checkpoint every 100 rows
    if i % 100 == 0 and i > 0:
        for j in range(i - 99, i + 1):  # loop over last 100
            notes_df.at[j, 'extracted_symptoms'] = json.dumps(extracted_list[j])
            notes_df.at[j, 'matched_hpo_terms'] = json.dumps(hpo_matches_list[j])

        notes_df.iloc[:i+1].to_csv('checkpoint_output.csv', index=False)
        print(f"💾 Checkpoint saved at row {i}")

# ---------- 📝 Add to DataFrame ----------
notes_df['extracted_symptoms'] = [json.dumps(x) for x in extracted_list]
notes_df['matched_hpo_terms'] = [json.dumps(x) for x in hpo_matches_list]

# ---------- 🧹 Split matched vs unmatched ----------
matched_ids_col = []
unmatched_symptoms_col = []

for row in hpo_matches_list:
    matched_ids = []
    unmatched = []

    for item in row:
        if isinstance(item, tuple) and len(item) == 2:
            symptom, hpo_id = item
            if hpo_id:
                matched_ids.append(hpo_id)
            else:
                unmatched.append(symptom)

    matched_ids_col.append(matched_ids)
    unmatched_symptoms_col.append(unmatched)

notes_df['matched_hpo_ids'] = matched_ids_col
notes_df['unmatched_symptoms'] = unmatched_symptoms_col

# ---------- 💾 Save final output ----------
notes_df.to_csv('Datasets/cleaned_dataset.csv', index=False)
print("✅ All done! Output saved to 'Datasets/cleaned_dataset.csv'")
