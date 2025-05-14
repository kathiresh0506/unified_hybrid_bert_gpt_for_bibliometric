import pandas as pd
import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import spacy
import ast

# Load models
print("🔄 Loading models...")
kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
nlp = spacy.load("en_core_sci_sm")  # SciSpacy for scientific NER

# Load data
df = pd.read_csv("papers.csv")
print(f"📄 Loaded {len(df)} papers.")

# ----------------------------
# STEP 1: CLEANING TEXT
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

df["clean_abstract"] = df["abstract"].fillna("").apply(clean_text)

# ----------------------------
# STEP 2: EXTRACT KEYWORDS
# ----------------------------
def extract_keywords(text):
    if not text or len(text) < 50:
        return []
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )
    return [kw[0] for kw in keywords]

print("🔍 Extracting keywords...")
df["keywords"] = df["clean_abstract"].apply(extract_keywords)

# ----------------------------
# STEP 3: ENTITY RECOGNITION (SciSpacy)
# ----------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
    return list(set(entities))  # remove duplicates

print("🔬 Extracting scientific entities...")
df["entities"] = df["clean_abstract"].apply(extract_entities)

# ----------------------------
# STEP 4: SAVE ENRICHED DATA
# ----------------------------
df.to_csv("papers_enriched.csv", index=False)
print("✅ Enriched metadata saved to papers_enriched.csv")

# Optional: Preview
print("\n🔹 Sample enriched record:")
print(df[["title", "keywords", "entities"]].head(1).to_string(index=False))
