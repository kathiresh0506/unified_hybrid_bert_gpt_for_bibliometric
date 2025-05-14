from py2neo import Graph

# Connect to your Neo4j DB
graph = Graph("bolt://localhost:7687", auth=("neo4j", "qwertyuiop"))

def get_graph_info(paper_title):
    query = f"""
    MATCH (p:Paper {{title: $title}})
    OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
    RETURN 
        collect(DISTINCT a.name) AS authors,
        collect(DISTINCT k.name) AS keywords,
        collect(DISTINCT e.name) AS entities
    """
    result = graph.run(query, title=paper_title).data()
    if result:
        info = result[0]
        return {
            "authors": info["authors"] or [],
            "keywords": info["keywords"] or [],
            "entities": info["entities"] or []
        }
    return {"authors": [], "keywords": [], "entities": []}
