import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from keybert import KeyBERT
from py2neo import Graph, Node, Relationship
from sklearn.mixture import GaussianMixture
import openai
import json, torch, ast, re
from fetch_papers import fetch_openalex_papers, save_to_csv  # Replace with actual functions

# === Global Model Loaders ===
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model='bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# === Config ===
openai.api_key = " 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"
openai.api_base = "https://api.groq.com/openai/v1"
graph = Graph("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))


# === Pipeline Stages ===
def fetch_and_save_papers(query, max_results=50):
    df = fetch_openalex_papers(query, max_results=max_results)
    save_to_csv(df, "papers.csv")


def enrich_papers():
    import ast
    import re
    df = pd.read_csv("papers.csv")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()

    df["clean_abstract"] = df["abstract"].fillna("").apply(clean_text)

    def safe_eval(val):
        if pd.isna(val):
            return []
        try:
            val = str(val).replace("‐", "-")  # Replace non-ASCII hyphen
            return ast.literal_eval(val)
        except Exception:
            return []

    # Handle missing or malformed columns
    df["keywords"] = df["keywords"].apply(safe_eval) if "keywords" in df.columns else [[] for _ in range(len(df))]
    df["entities"] = df["entities"].apply(safe_eval) if "entities" in df.columns else [[] for _ in range(len(df))]

    df.to_csv("papers_enriched.csv", index=False)




def build_knowledge_graph():
    df = pd.read_csv("papers_enriched.csv")
    graph.delete_all()

    for _, row in df.iterrows():
        paper_node = Node("Paper", title=row["title"], year=row["publication_year"], doi=row["doi"])
        graph.create(paper_node)

        for author in eval(str(row["authors"])):
            author_node = Node("Author", name=author)
            graph.merge(author_node, "Author", "name")
            graph.create(Relationship(author_node, "WROTE", paper_node))

        for keyword in row["keywords"]:
            keyword_node = Node("Keyword", name=keyword)
            graph.merge(keyword_node, "Keyword", "name")
            graph.create(Relationship(paper_node, "HAS_KEYWORD", keyword_node))

        for entity in row["entities"]:
            entity_node = Node("Entity", name=entity)
            graph.merge(entity_node, "Entity", "name")
            graph.create(Relationship(paper_node, "MENTIONS", entity_node))


def cluster_abstracts():
    df = pd.read_csv("papers_enriched.csv").dropna(subset=["clean_abstract"])
    embeddings = model.encode(df["clean_abstract"].tolist(), show_progress_bar=True)
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    df["gmm_cluster"] = gmm.predict(embeddings)
    df["gmm_probs"] = gmm.predict_proba(embeddings).tolist()
    df.to_csv("clustered_papers.csv", index=False)


def summarize_clusters():
    df = pd.read_csv("clustered_papers.csv")
    df.columns = df.columns.str.strip()
    clusters = df.groupby("gmm_cluster")["clean_abstract"].apply(list).to_dict()

    def embedding_to_keywords(texts):
        combined_text = " ".join(texts)
        return [kw[0] for kw in kw_model.extract_keywords(combined_text, top_n=20)]

    def query_groq(prompt):
        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response["choices"][0]["message"]["content"]

    summaries = {}
    for cluster_id, texts in clusters.items():
        try:
            keywords = embedding_to_keywords(texts)
            prompt = (
                "You are a research assistant. Generate a detailed research summary "
                "focusing on the research methodology and key findings based on the following concepts:\n\n"
                + ", ".join(keywords) + "\n\nSummary:"
            )
            summaries[cluster_id] = query_groq(prompt)
        except Exception as e:
            summaries[cluster_id] = f"ERROR: {str(e)}"

    with open("cluster_summaries.json", "w") as f:
        json.dump(summaries, f, indent=4)


# === Combined Pipeline ===
def run_pipeline(query):
    try:
        fetch_and_save_papers(query)
        enrich_papers()
        build_knowledge_graph()
        cluster_abstracts()
        summarize_clusters()
        return "✅ Pipeline executed successfully!"
    except Exception as e:
        return f"❌ Pipeline failed: {str(e)}"


# === STREAMLIT UI ===
st.set_page_config(layout="wide", page_title="Unified Hybrid NLP System")

st.title("📚 Unified Hybrid NLP Research Assistant")
st.markdown("End-to-end pipeline from OpenAlex → KG → Clustering → Groq summarization")

query = st.text_input("Enter your research topic:", "large language models in medicine")
run = st.button("Run Pipeline")

if run and query:
    with st.spinner("Running pipeline..."):
        status = run_pipeline(query)

    st.success(status)

    if "successfully" in status:
        try:
            df1 = pd.read_csv("papers_enriched.csv")
            df2 = pd.read_csv("clustered_papers.csv")
            with open("cluster_summaries.json") as f:
                summaries = json.load(f)

            st.subheader("📄 Enriched Paper Data")
            st.dataframe(df1.head(10))

            st.subheader("📄 Clustered Papers")
            st.dataframe(df2[["title", "gmm_cluster"]].head(10))

            st.subheader("📊 Cluster Summaries")
            for cluster_id, summary in summaries.items():
                st.markdown(f"**🔹 Cluster {cluster_id}**")
                st.write(summary)

            with open("papers_enriched.csv", "rb") as f:
                st.download_button("📥 Download Enriched CSV", f, file_name="papers_enriched.csv")

            with open("clustered_papers.csv", "rb") as f:
                st.download_button("📥 Download Clustered CSV", f, file_name="clustered_papers.csv")

            with open("cluster_summaries.json", "rb") as f:
                st.download_button("📥 Download Cluster Summaries", f, file_name="cluster_summaries.json")

        except Exception as e:
            st.error(f"Error displaying results: {e}")
