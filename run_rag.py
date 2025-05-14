from rag_retrieve import retrieve
from rag_generate import generate_summary

query = input("🔍 Enter your research topic: ")
docs = retrieve(query, k=5)

print("\n📚 Top Documents:\n")
for doc in docs:
    print(doc)

print("\n🧠 Summary:\n")
print(generate_summary(docs))
