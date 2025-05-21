# fetch_papers.py
import requests
import pandas as pd

def fetch_core_papers(query, max_results=10):
    api_key = "ZLQojgG1uJDYRdprWS8UhEzsIPM03cNi"  # 🔁 Replace with your real key
    url = f"https://core.ac.uk:443/api-v2/search/{query}?page=1&pageSize={max_results}&metadata=true&fulltext=true&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"CORE API request failed: {response.status_code}")
    
    results = response.json()["data"]
    papers = []
    for item in results:
        papers.append({
            "title": item.get("title", ""),
            "authors": [a.get("name", "") for a in item.get("authors", [])] if "authors" in item else [],
            "abstract": item.get("description", ""),
            "full_text": item.get("fullText", ""),
            "publication_year": item.get("publishedDate", "")[:4] if item.get("publishedDate") else "",
            "doi": item.get("doi", ""),
            "keywords": item.get("topics", [])
        })
    return pd.DataFrame(papers)

def save_to_csv(df, filename="papers2.csv"):
    df.to_csv(filename, index=False)
