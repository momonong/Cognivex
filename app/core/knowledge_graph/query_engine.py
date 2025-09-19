import json
from typing import List

from app.services.neo4j_connector import get_neo4j_driver

driver = get_neo4j_driver()

CYPHER_TEMPLATE = """
UNWIND $regionNames AS regionName
MATCH (r:Region {name: regionName})
OPTIONAL MATCH (r)-[:BELONGS_TO]->(n:YeoNetwork)
OPTIONAL MATCH (r)-[:HAS_FUNCTION]->(f:Function)
RETURN
  r.name AS region,
  n.name AS network,
  COLLECT(DISTINCT f.name) AS functions,
  r.ad_associated AS isADAssociated
"""

def graphrag(region_names: List[str]) -> str:
    """
    Receives a CLEAN, VALIDATED list of brain region names and executes a precise,
    templated Cypher query to get their associated data from the knowledge graph.
    The input MUST be a list of strings that are guaranteed to exist in the database.
    """
    if not isinstance(region_names, list) or not region_names:
        return json.dumps({"result": [], "message": "Input region list was empty."})

    # print(f"--- [Tool Log] Running stable Cypher query for {region_names} regions... ---")
    try:
        with driver.session() as session:
            result = session.run(CYPHER_TEMPLATE, {"regionNames": region_names})
            records = [record.data() for record in result]
        return json.dumps({"result": records}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Database query failed: {str(e)}"})