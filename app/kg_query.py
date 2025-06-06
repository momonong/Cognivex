from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class BrainKG:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="NEO4J_PASSWORD"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_region_function(self, region_name: str) -> list[str]:
        query = """
        MATCH (r:Region {name: $region_name})-[:HAS_FUNCTION]->(f:Function)
        RETURN f.name AS function
        """
        with self.driver.session() as session:
            result = session.run(query, region_name=region_name)
            return [record["function"] for record in result]
