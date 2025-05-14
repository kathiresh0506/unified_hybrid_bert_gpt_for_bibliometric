import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("papers_enriched.csv")
index = faiss.read_index("semantic_index.faiss")
with open("semantic_titles.pkl", "rb") as f:
    titles = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=5):
    vec = model.encode([query])
    D, I = index.search(np.array(vec), k)

    results = []
    for idx in I[0]:
        title = df.iloc[idx]["title"]
        abstract = df.iloc[idx]["clean_abstract"]
        results.append(f"📄 **{title}**\n{abstract}\n")
    return results
