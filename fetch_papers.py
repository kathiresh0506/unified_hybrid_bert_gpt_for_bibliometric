
import requests
import pandas as pd
import time
import urllib.parse

def fetch_openalex_papers(query: str, max_results: int = 50) -> pd.DataFrame:
    print(f"Searching OpenAlex for: {query}")
    base_url = "https://api.openalex.org/works"
    per_page = 25
    all_results = []
    cursor = "*"

    # ✅ Real, properly formatted User-Agent
    headers = {
        "User-Agent": "LLMResearchBot/1.0 (mailto:your.email@example.com)"
    }

    encoded_query = urllib.parse.quote(query)

    while len(all_results) < max_results:
        url = (
            f"{base_url}?filter=title.search:{encoded_query}"
            f"&per-page={per_page}&cursor={cursor}"
        )

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

            # Rebuild abstract from inverted index
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



def save_to_csv(df: pd.DataFrame, filename: str = "papers.csv"):
    df.to_csv(filename, index=False)
    print(f"Saved metadata to {filename}")


if __name__ == "__main__":
    query = input("Enter your research topic: ")
    df = fetch_openalex_papers(query, max_results=50)
    save_to_csv(df)
