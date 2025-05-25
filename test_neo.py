from neo4j import GraphDatabase

URI = "neo4j+s://398c4cbb.databases.neo4j.io"  # cloud DB URI
USERNAME = "neo4j"
PASSWORD = "MoRiYhLULubPW3Q722MLGx8Eq4ScSyOScHX1N348kaM"  # from AuraDB

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

try:
    with driver.session() as session:
        result = session.run("RETURN 'AuraDB connected!' AS message")
        print(result.single()["message"])
except Exception as e:
    print("Connection failed:", e)
