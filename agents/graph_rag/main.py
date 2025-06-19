from .neo_client import run_cypher_query
from .query import query_gemini_with_graph

if __name__ == "__main__":
    question = "Which brain regions are related to Alzheimer's disease and what are their functions?"
    
    cypher = """
    MATCH (r:Region)-[:BELONGS_TO]->(n:Network)
    WHERE r.AD_Associated = 'Yes'
    RETURN r.name AS Region, r.Function AS RegionFunction, n.name AS Network
    """

    graph_data = run_cypher_query(cypher)
    answer = query_gemini_with_graph(question, graph_data)

    print("\n🧠 Answer from Gemini:")
    print(answer)
