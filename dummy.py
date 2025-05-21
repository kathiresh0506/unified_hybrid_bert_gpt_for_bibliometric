import requests

API_KEY = "ZLQojgG1uJDYRdprWS8UhEzsIPM03cNi"  # Replace with your key

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

search_url = "https://api.core.ac.uk/v2/search"

payload = {
    "query": "deep learning",
    "page": 1,
    "pageSize": 5,
    "metadata": True,
    "fulltext": True
}

response = requests.post(search_url, headers=headers, json=payload)

print("Status code:", response.status_code)
print("Response text:", response.text[:500])
