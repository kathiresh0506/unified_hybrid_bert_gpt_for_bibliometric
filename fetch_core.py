import requests
import pandas as pd
from keybert import KeyBERT
import fitz  # PyMuPDF for PDF parsing
import os

# ========= CONFIG =========
API_KEY = 'ZLQojgG1uJDYRdprWS8UhEzsIPM03cNi'  # Replace with your actual CORE API key
CORE_API_ENDPOINT = "https://api.core.ac.uk/v3/search/works"
kw_model = KeyBERT()

# ========= Keyword Refinement =========
def refine_keywords(keywords):
    replacements = {
        "ai": "artificial intelligence",
        "acquire": "language acquisition",
        "early childhood": "child language development",
        "medical imaging": "medical image analysis"
    }
    refined = set()
    for phrase in keywords:
        phrase = phrase.lower()
        matched = False
        for key, val in replacements.items():
            if key in phrase:
                refined.add(val)
                matched = True
                break
        if not matched:
            refined.add(phrase)
    return list(refined)

# ========= Extract Keywords from NL Query =========
def process_nl_query(nl_query, top_n=5):
    raw_keywords = kw_model.extract_keywords(
        nl_query,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        use_mmr=True,
        diversity=0.7,
        nr_candidates=30,
        top_n=top_n
    )
    phrases = [kw[0] for kw in raw_keywords]
    print("🔑 Raw Keywords:", phrases)
    refined = refine_keywords(phrases)
    print("🔧 Refined Keywords:", refined)
    return ' OR '.join(f'"{word}"' for word in refined)

# ========= CORE API Search =========
def query_core_api(keyword_query, max_results=10):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    params = {'q': keyword_query, 'limit': max_results}
    response = requests.get(CORE_API_ENDPOINT, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        print("❌ Error:", response.status_code, response.text)
        return []

# ========= PDF Text Extractor =========
def extract_text_from_pdf_link(link):
    try:
        response = requests.get(link)
        if response.status_code == 200:
            pdf_path = "temp.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            with fitz.open(pdf_path) as doc:
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
            os.remove(pdf_path)
            return full_text.strip()
    except Exception as e:
        print(f"⚠️ PDF extraction failed: {e}")
    return None

# ========= Download Full Texts =========
def get_full_texts(papers):
    full_texts = []
    for paper in papers:
        title = paper.get('title', 'No Title')
        link = paper.get('fullTextLink')
        print(f"\n📥 Fetching PDF: {title}")
        text = extract_text_from_pdf_link(link) if link else None
        full_texts.append({
            'title': title,
            'authors': paper.get('authors', []),
            'year': paper.get('yearPublished'),
            'doi': paper.get('doi'),
            'abstract': paper.get('abstract', ''),
            'fullText': text if text else "Not available",
            'link': link
        })
    return full_texts

# ========= Save to CSV =========
def save_papers_to_csv(data, filename="core_papers_fulltext.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n✅ Saved to {filename}")

# ========= MAIN =========
def process_query_and_fetch_papers(nl_query, max_results=10):
    print("\n🔍 Natural Language Query:", nl_query)
    keyword_query = process_nl_query(nl_query)
    print("🎯 CORE API Query:", keyword_query)
    papers = query_core_api(keyword_query, max_results=max_results)
    if not papers:
        print("❌ No papers found.")
        return
    results = get_full_texts(papers)
    save_papers_to_csv(results)

# ========= Run =========
if __name__ == "__main__":
    query = input("Enter your research topic: ")
    process_query_and_fetch_papers(query, max_results=10)
