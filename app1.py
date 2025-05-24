import streamlit as st
import pandas as pd
import json

from pathlib import Path
from your_pipeline import (
    fetch_and_save_papers,
    enrich_papers,
    build_knowledge_graph,
    cluster_texts,
    summarize_clusters,
    visualize_graph,
    print_enriched_papers,
    print_clusters,
    print_summaries
)

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="Bibliometric NLP Assistant", layout="wide")

st.title("📚 Bibliometric Research Assistant")
st.markdown("Enter a research query below and explore bibliometric insights extracted from arXiv papers.")

# === USER INPUT ===
query = st.text_input("🔍 Enter your research query:", "")

if st.button("🚀 Run Pipeline") and query:
    with st.spinner("Running the full bibliometric pipeline..."):
        fetch_and_save_papers(query)
        enrich_papers()
        build_knowledge_graph()
        cluster_texts()
        summarize_clusters()
        visualize_graph()
    st.success("✅ Pipeline completed!")

    # === DISPLAY RESULTS ===
    if Path("papers_enriched.csv").exists():
        df = pd.read_csv("papers_enriched.csv")
        st.subheader("📄 Top Enriched Papers with Keywords")
        for idx, row in df.head(5).iterrows():
            st.markdown(f"""
            ### 📘 {row['title']}
            - 🗓️ Year: {row['year']}
            - 🔑 Keywords: {', '.join(eval(row['keywords']))}
            """)

    if Path("clustered_papers.csv").exists():
        df_cluster = pd.read_csv("clustered_papers.csv")
        st.subheader("🧠 Papers by Cluster")
        for cluster in sorted(df_cluster['cluster'].unique()):
            st.markdown(f"#### 🧩 Cluster {cluster}")
            for _, row in df_cluster[df_cluster["cluster"] == cluster].head(3).iterrows():
                st.markdown(f"- {row['title']}")

    if Path("cluster_summaries.json").exists():
        with open("cluster_summaries.json", "r", encoding="utf-8") as f:
            summaries = json.load(f)
        st.subheader("📝 Cluster Summaries")
        for cid, summary in summaries.items():
            st.markdown(f"**Cluster {cid}:**")
            st.markdown(summary)

    if Path("graph.html").exists():
        st.subheader("🌐 Knowledge Graph Visualization")
        with open("graph.html", "r", encoding="utf-8") as f:
            graph_html = f.read()
        st.components.v1.html(graph_html, height=600, scrolling=True)

# Optional footer
st.markdown("---")
st.markdown("Made with ❤️ using arXiv, BERT, GROQ, and Neo4j.")
