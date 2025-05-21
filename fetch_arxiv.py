import requests
import feedparser
import time
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd

def fetch_arxiv_papers(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(base_url, params=params)
    feed = feedparser.parse(response.content)

    papers = []
    for entry in feed.entries:
        pdf_url = entry.id.replace("abs", "pdf") + ".pdf"
        try:
            pdf_resp = requests.get(pdf_url)
            reader = PdfReader(BytesIO(pdf_resp.content))
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            full_text = ""
        paper = {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "year": entry.published[:4],
            "doi": entry.id,
            "full_text": full_text
        }
        papers.append(paper)
        time.sleep(3)  # Avoid hitting arXiv too fast
    return pd.DataFrame(papers)

def save_to_csv(df, path="papers3.csv"):
    df.to_csv(path, index=False)
