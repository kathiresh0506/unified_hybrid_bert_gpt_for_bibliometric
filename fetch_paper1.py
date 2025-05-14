import requests

# Replace with your actual API key
API_KEY = "ZLQojgG1uJDYRdprWS8UhEzsIPM03cNi"
BASE_URL = "https://api.core.ac.uk/v3/search/works"

# Your search query
query = "natural language processing"

# Parameters for the API request
params = {
    "q": query,          # search query
    "page": 1,           # pagination (start from page 1)
    "pageSize": 10,      # number of results per page
    "apiKey": API_KEY    # your CORE API key
}

# Send the GET request
response = requests.get(BASE_URL, params=params)

# Check if the request was successful
if response.status_code == 200:
    results = response.json()
    
    # Loop through each article in the results
    for i, article in enumerate(results.get("data", []), 1):
        print(f"\nArticle {i}")
        print("Title:", article.get("title"))
        print("Authors:", article.get("authors"))
        print("Year:", article.get("year"))
        print("DOI:", article.get("doi"))
        print("Download URL:", article.get("downloadUrl"))
        print("Full Text Available:", article.get("fullText", False))
else:
    print("Failed to retrieve articles:", response.status_code, response.text)
