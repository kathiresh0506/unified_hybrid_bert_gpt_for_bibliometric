from py2neo import Graph, Node, Relationship
import pandas as pd
import ast

# Connect to local Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))
 # change password if needed

# Load enriched paper data
df = pd.read_csv("papers_enriched.csv")

# Clear existing data (optional, for development)
graph.delete_all()

print("📥 Inserting nodes into Neo4j...")

for _, row in df.iterrows():
    paper_node = Node("Paper", title=row["title"], year=row["publication_year"], doi=row["doi"])
    graph.create(paper_node)

    # Authors
    if pd.notna(row["authors"]):
        for author_name in row["authors"].split(","):
            author_name = author_name.strip()
            if not author_name: continue
            author_node = Node("Author", name=author_name)
            graph.merge(author_node, "Author", "name")
            graph.create(Relationship(author_node, "WROTE", paper_node))

    # Keywords
    try:
        keywords = ast.literal_eval(row["keywords"])
        for keyword in keywords:
            keyword_node = Node("Keyword", name=keyword)
            graph.merge(keyword_node, "Keyword", "name")
            graph.create(Relationship(paper_node, "HAS_KEYWORD", keyword_node))
    except:
        pass

    # Entities (scientific NER)
    try:
        entities = ast.literal_eval(row["entities"])
        for ent in entities:
            ent_node = Node("Entity", name=ent)
            graph.merge(ent_node, "Entity", "name")
            graph.create(Relationship(paper_node, "MENTIONS", ent_node))
    except:
        pass

print("✅ Knowledge Graph built successfully in Neo4j.")
