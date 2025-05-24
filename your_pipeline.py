import pandas as pd
import re, json, ast, torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from keybert import KeyBERT
from py2neo import Graph, Node, Relationship
from sklearn.mixture import GaussianMixture
from pyvis.network import Network
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("GROK_API_KEY")
print("API KEY:",API_KEY)
# === MODEL SETUP ===
bert_encoder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model='bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# === GROQ API SETUP ===
client = OpenAI(
    api_key= API_KEY,  
    base_url="https://api.groq.com/openai/v1"
)

# === NEO4J SETUP ===
graph = Graph("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))

# === FETCHING ===
def fetch_and_save_papers(query, max_results=10):
    from fetch_arxiv import fetch_arxiv_papers  # make sure you have this file
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
    graph.delete_all()
    for _, row in df.iterrows():
        paper_node = Node("Paper", title=row["title"], year=row["year"], doi=row["doi"])
        graph.create(paper_node)

        for author in ast.literal_eval(row["authors"]):
            author_node = Node("Author", name=author)
            graph.merge(author_node, "Author", "name")
            graph.create(Relationship(author_node, "WROTE", paper_node))

        for keyword in ast.literal_eval(row["keywords"]):
            keyword_node = Node("Keyword", name=keyword)
            graph.merge(keyword_node, "Keyword", "name")
            graph.create(Relationship(paper_node, "HAS_KEYWORD", keyword_node))

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
    query = """
    MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
    RETURN p.title AS title, k.name AS keyword LIMIT 100
    """
    results = graph.run(query).data()
    net = Network(notebook=False, cdn_resources='in_line')
    for record in results:
        net.add_node(record["title"], label=record["title"], shape="box", color="lightblue")
        net.add_node(record["keyword"], label=record["keyword"], shape="dot", color="orange")
        net.add_edge(record["title"], record["keyword"])

    html_content = net.generate_html()
    with open("graph.html", "w", encoding="utf-8") as f:
        f.write(html_content)

# === DISPLAY HELPERS (OPTIONAL) ===
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
