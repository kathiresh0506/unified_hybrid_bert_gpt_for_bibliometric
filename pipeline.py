import requests
import pandas as pd
import time
import urllib.parse
import re
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from py2neo import Graph, Node, Relationship
from sklearn.mixture import GaussianMixture
import ast
import json
import openai
from transformers import BertTokenizer, BertModel
import torch
import gradio as gr

# --- Fetch Papers from OpenAlex ---
def fetch_openalex_papers(query: str, max_results: int = 50) -> pd.DataFrame:
    print(f"Searching OpenAlex for: {query}")
    base_url = "https://api.openalex.org/works"
    per_page = 25
    all_results = []
    cursor = "*"

    headers = {
        "User-Agent": "LLMResearchBot/1.0 (mailto:your.email@example.com)"
    }

    encoded_query = urllib.parse.quote(query)

    while len(all_results) < max_results:
        url = f"{base_url}?filter=title.search:{encoded_query}&per-page={per_page}&cursor={cursor}"

        try:
            response = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

        if response.status_code == 403:
            print("❌ Access Forbidden (403) — Check your User-Agent and network.")
            break
        elif response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        data = response.json()
        results = data.get("results", [])

        for work in results:
            paper = {
                "id": work.get("id"),
                "title": work.get("title"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "authors": ", ".join([a['author']['display_name'] for a in work.get("authorships", [])]),
                "abstract": work.get("abstract_inverted_index", None),
                "open_access": work.get("open_access", {}).get("is_oa", False),
                "host_venue": work.get("host_venue", {}).get("display_name", "")
            }

            if paper["abstract"]:
                abstract_words = sorted(
                    [(v, k) for k, vlist in paper["abstract"].items() for v in vlist]
                )
                paper["abstract"] = " ".join([w for _, w in abstract_words])
            else:
                paper["abstract"] = ""

            all_results.append(paper)

        cursor = data.get('meta', {}).get('next_cursor', None)
        if not cursor:
            break

        time.sleep(1)  # rate-limiting

    df = pd.DataFrame(all_results)
    print(f"✅ Retrieved {len(df)} papers.")
    return df

# --- Preprocessing ---
def preprocess_data(df: pd.DataFrame):
    kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
    nlp = spacy.load("en_core_sci_sm")  # SciSpacy for scientific NER

    # Cleaning text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()

    df["clean_abstract"] = df["abstract"].fillna("").apply(clean_text)

    # Extracting keywords
    def extract_keywords(text):
        if not text or len(text) < 50:
            return []
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)
        return [kw[0] for kw in keywords]

    df["keywords"] = df["clean_abstract"].apply(extract_keywords)

    # Entity Recognition (SciSpacy)
    def extract_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
        return list(set(entities))

    df["entities"] = df["clean_abstract"].apply(extract_entities)

    df.to_csv("papers_enriched.csv", index=False)
    print("✅ Enriched metadata saved to papers_enriched.csv")

    return df

# --- Building FAISS Index ---
def build_faiss_index(df: pd.DataFrame):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    abstracts = df["clean_abstract"].fillna("").tolist()
    embeddings = model.encode(abstracts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, "semantic_index.faiss")
    with open("semantic_titles.pkl", "wb") as f:
        pickle.dump(df["title"].tolist(), f)

    print("✅ FAISS index and titles saved.")
    return index

# --- Neo4j Knowledge Graph ---
def build_neo4j_graph(df: pd.DataFrame):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))

    graph.delete_all()  # Optional, for development

    print("📥 Inserting nodes into Neo4j...")
    for _, row in df.iterrows():
        paper_node = Node("Paper", title=row["title"], year=row["publication_year"], doi=row["doi"])
        graph.create(paper_node)

        if pd.notna(row["authors"]):
            for author_name in row["authors"].split(","):
                author_name = author_name.strip()
                if not author_name: continue
                author_node = Node("Author", name=author_name)
                graph.merge(author_node, "Author", "name")
                graph.create(Relationship(author_node, "WROTE", paper_node))

        try:
            keywords = ast.literal_eval(row["keywords"])
            for keyword in keywords:
                keyword_node = Node("Keyword", name=keyword)
                graph.merge(keyword_node, "Keyword", "name")
                graph.create(Relationship(paper_node, "HAS_KEYWORD", keyword_node))
        except:
            pass

        try:
            entities = ast.literal_eval(row["entities"])
            for ent in entities:
                ent_node = Node("Entity", name=ent)
                graph.merge(ent_node, "Entity", "name")
                graph.create(Relationship(paper_node, "MENTIONS", ent_node))
        except:
            pass

    print("✅ Knowledge Graph built successfully in Neo4j.")

# --- Clustering with GMM ---
def perform_clustering(df: pd.DataFrame):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["clean_abstract"].dropna().tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(embeddings)

    df["gmm_cluster"] = gmm.predict(embeddings)
    df["gmm_probs"] = gmm.predict_proba(embeddings).tolist()

    df.to_csv("clustered_papers.csv", index=False)
    print("✅ Clustered data saved to clustered_papers.csv")

# --- Groq Summarization ---
def summarize_with_groq(file_path):
    # Load and Group Clusters
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean any extra spaces in column names
    clusters = df.groupby("gmm_cluster")["clean_abstract"].apply(list).to_dict()

    # Groq API Setup
    openai.api_key = "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"
    openai.api_base = "https://api.groq.com/openai/v1"

    # Load Models
    kw_model = KeyBERT(model='bert-base-uncased')

    # Summarize each cluster
    all_summaries = {}

    def embedding_to_keywords(texts, num_keywords=10):
        combined_text = " ".join(texts)
        keywords = kw_model.extract_keywords(combined_text, top_n=20)
        return [kw[0] for kw in keywords]

    def build_prompt(keywords):
        return (
            "You are a research assistant. Generate a detailed research summary "
            "focusing on the research methodology and key findings based on the following concepts:\n\n"
            + ", ".join(keywords) + "\n\nSummary:"
        )

    def query_groq(prompt):
        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response["choices"][0]["message"]["content"]

    def process_cluster(texts):
        try:
            keywords = embedding_to_keywords(texts)
            prompt = build_prompt(keywords)
            return query_groq(prompt)
        except Exception as e:
            print(f"⚠️ Error processing cluster: {e}")
            return "ERROR"

    # Process all clusters
    for label, texts in clusters.items():
        print(f"\n🔹 Cluster {label} 🔹")
        try:
            summary = process_cluster(texts)
            all_summaries[label] = summary
            print(summary)
        except Exception as e:
            print(f"⚠️ Error processing cluster {label}: {e}")
            all_summaries[label] = "ERROR"

    # Save summaries
    with open("cluster_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=4)

    print("\n✅ All summaries saved to 'cluster_summaries.json'")

# --- Gradio Interface ---
def run_pipeline(query: str):
    # Step 1: Fetch Papers
    df = fetch_openalex_papers(query)
    df.to_csv("papers.csv", index=False)

    # Step 2: Preprocess Data
    df_enriched = preprocess_data(df)

    # Step 3: Build FAISS Index
    build_faiss_index(df_enriched)

    # Step 4: Build Knowledge Graph in Neo4j
    build_neo4j_graph(df_enriched)

    # Step 5: Perform Clustering
    perform_clustering(df_enriched)

    # Step 6: Summarize Clusters with Groq
    summarize_with_groq("clustered_papers.csv")

    return "Pipeline executed successfully!"

# Gradio Interface
gr.Interface(fn=run_pipeline, 
             inputs=gr.Textbox(label="Research Topic", placeholder="Enter your research topic here..."), 
             outputs="text", 
             title="Unified Research Paper Pipeline").launch()
