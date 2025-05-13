from keybert import KeyBERT
import requests

# ========== CONFIG ==========
API_KEY = 'ZLQojgG1uJDYRdprWS8UhEzsIPM03cNi'  # <-- Replace with your CORE API key
CORE_API_ENDPOINT = "https://api.core.ac.uk/v3/search/works"

# ========== Initialize KeyBERT ==========
kw_model = KeyBERT()

# ========== Academic Synonym Mapping ==========
def refine_keywords(keywords):
    replacements = {
        "ai": "artificial intelligence",
        "acquire": "language acquisition",
        "acquire language": "language acquisition",
        "early childhood": "child language development",
        "medical imaging": "medical image analysis",
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

# ========== Convert NL Query to Academic Keyword Query ==========
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

# ========== Query CORE API ==========
def query_core_api(keyword_query, max_results=10):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    params = {
        'q': keyword_query,
        'limit': max_results
    }
    response = requests.get(CORE_API_ENDPOINT, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        print("❌ Error:", response.status_code, response.text)
        return []

# ========== Display Results ==========
def show_paper_results(papers):
    if not papers:
        print("\n🚫 No papers found. Try rephrasing your query.")
        return
    for idx, paper in enumerate(papers, 1):
        print(f"\n📄 Paper {idx}")
        print("Title:", paper.get('title'))
        print("Authors:", paper.get('authors', []))
        print("Year:", paper.get('yearPublished'))
        print("DOI:", paper.get('doi'))
        print("Full Text:", paper.get('fullTextLink'))
        print("Abstract:", paper.get('abstract') or "No abstract available")
        print("-" * 50)

# ========== MAIN PIPELINE ==========
def process_query_and_fetch_papers(nl_query):
    print("\n🔍 Natural Language Query:", nl_query)
    keyword_query = process_nl_query(nl_query)
    print("🎯 CORE API Keyword Query:", keyword_query)
    papers = query_core_api(keyword_query)
    show_paper_results(papers)

# ========== Example ==========
if __name__ == "__main__":
    nl_query =input("Enter the query:")
    process_query_and_fetch_papers(nl_query)
