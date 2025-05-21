# fetch_paper.py

import requests
import pandas as pd
import os

CORE_API_KEY = "ZLQojgG1uJDYRdprWS8UhEzsIPM03cNi"  # 🔁 Replace this with your actual CORE API key

def fetch_core_papers(query, max_results=10):
    """
    Fetches academic papers from CORE based on a search query.

    Args:
        query (str): The search term to query CORE with.
        max_results (int): Maximum number of papers to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the paper metadata.
    """
    url = "https://core.ac.uk:443/api-v2/articles/search"
    params = {
        "q": query,
        "page": 1,
        "pageSize": max_results,
        "metadata": True,
        "fulltext": True,
        "apiKey": CORE_API_KEY,
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

    records = []
    for paper in data.get("data", []):
        records.append({
            "title": paper.get("title", ""),
            "authors": [author.get("name", "") for author in paper.get("authors", [])] if paper.get("authors") else [],
            "full_text": paper.get("fullText", "") or paper.get("description", ""),
            "year": paper.get("publishedDate", "")[:4] if paper.get("publishedDate") else "",
            "doi": paper.get("doi", ""),
            "url": paper.get("downloadUrl") or paper.get("identifier", ""),
        })

    return pd.DataFrame(records)

def save_to_csv(df, filename):
    """
    Saves the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Output CSV file path.
    """
    df.to_csv(filename, index=False)