import pandas as pd
import re
import json
import ast
import os
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.mixture import GaussianMixture
from openai import OpenAI
from neo4j import GraphDatabase

# === HARDCODED SECRETS ===
API_KEY = "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"
NEO4J_URI = "neo4j+s://398c4cbb.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "MoRiYhLULubPW3Q722MLGx8Eq4ScSyOScHX1N348kaM"

# === MODEL SETUP ===
bert_encoder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model='bert-base-uncased')

# === GROQ API SETUP ===
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# === NEO4J DRIVER SETUP ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === FETCHING ===
def fetch_and_save_papers(query, max_results=10):
    from fetch_arxiv import fetch_arxiv_papers
    df = fetch_arxiv_papers(query, max_results=max_results)
    df = df.dropna(subset=["full_text"])
    df.to_csv("papers3.csv", index=False)

# === ENRICH ===
def enrich_papers():
    df = pd.read_csv("papers3.csv")

    def clean_text(text):
        text = str(text).lower()
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

    df["clean_text"] = df["full_text"].fillna("").apply(clean_text)
    df["keywords"] = df["clean_text"].apply(
        lambda x: [kw[0] for kw in kw_model.extract_keywords(x, top_n=10)] if x else []
    )
    df.to_csv("papers_enriched.csv", index=False)

# === GRAPH ===
def build_knowledge_graph():
    df = pd.read_csv("papers_enriched.csv")
    records = df.to_dict("records")

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for row in records:
            title = row.get("title")
            year = row.get("year")
            doi = row.get("doi")
            try:
                authors = ast.literal_eval(row.get("authors", "[]"))
            except Exception:
                authors = []
            try:
                keywords = ast.literal_eval(row.get("keywords", "[]"))
            except Exception:
                keywords = []

            session.run(
                """
                MERGE (p:Paper {title: $title})
                SET p.year = $year, p.doi = $doi
                """,
                title=title, year=year, doi=doi
            )

            for author in authors:
                session.run(
                    """
                    MERGE (a:Author {name: $author})
                    MERGE (p:Paper {title: $title})
                    MERGE (a)-[:WROTE]->(p)
                    """,
                    author=author,
                    title=title
                )

            for keyword in keywords:
                session.run(
                    """
                    MERGE (k:Keyword {name: $keyword})
                    MERGE (p:Paper {title: $title})
                    MERGE (p)-[:HAS_KEYWORD]->(k)
                    """,
                    keyword=keyword,
                    title=title
                )

# === CLUSTER ===
def cluster_texts():
    df = pd.read_csv("papers_enriched.csv").dropna(subset=["clean_text"])
    embeddings = bert_encoder.encode(df["clean_text"].tolist(), show_progress_bar=True)
    gmm = GaussianMixture(n_components=5, random_state=42)
    df["cluster"] = gmm.fit_predict(embeddings)
    df.to_csv("clustered_papers.csv", index=False)

# === SUMMARIZE ===
def summarize_clusters():
    df = pd.read_csv("clustered_papers.csv")
    grouped = df.groupby("cluster")["clean_text"].apply(list).to_dict()

    def query_groq(prompt):
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a bibliometric research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )
        return response.choices[0].message.content

    summaries = {}
    for cluster, texts in grouped.items():
        text_blob = " ".join(texts)[:4000]
        prompt = (
            "Read the following full research texts and provide a scholarly summary of their bibliometric insights "
            "including methods, findings, and trends:\n" + text_blob
        )
        summaries[cluster] = query_groq(prompt)

    with open("cluster_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

# === VISUALIZE ===
def visualize_graph():
    with driver.session() as session:
        query = """
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN p.title AS title, k.name AS keyword LIMIT 100
        """
        results = session.run(query).data()

    from pyvis.network import Network
    net = Network(notebook=False, cdn_resources='in_line')
    for record in results:
        net.add_node(record["title"], label=record["title"], shape="box", color="lightblue")
        net.add_node(record["keyword"], label=record["keyword"], shape="dot", color="orange")
        net.add_edge(record["title"], record["keyword"])

    html_content = net.generate_html()
    with open("graph.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# === DISPLAY HELPERS ===
def print_enriched_papers():
    df = pd.read_csv("papers_enriched.csv")
    for idx, row in df.head(5).iterrows():
        keywords = eval(row["keywords"]) if isinstance(row["keywords"], str) else row["keywords"]
        print(f"\n📘 {row['title']}\n🗓️ Year: {row['year']}")
        print(f"🔑 Keywords: {', '.join(keywords)}")

def print_clusters():
    df = pd.read_csv("clustered_papers.csv")
    for cluster in sorted(df['cluster'].unique()):
        print(f"\n--- Cluster {cluster} ---")
        papers = df[df['cluster'] == cluster].head(3)
        for _, row in papers.iterrows():
            print(f"• {row['title']}")

def print_summaries():
    with open("cluster_summaries.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)
    for cid, summary in summaries.items():
        print(f"\n--- 🧩 Cluster {cid} Summary ---\n{summary.strip()}\n")
