import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load enriched data
df = pd.read_csv("papers_enriched.csv")
abstracts = df["clean_abstract"].fillna("").tolist()
titles = df["title"].fillna("").tolist()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(abstracts, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and titles
faiss.write_index(index, "semantic_index.faiss")
with open("semantic_titles.pkl", "wb") as f:
    pickle.dump(titles, f)

print("✅ FAISS index and titles saved.")
