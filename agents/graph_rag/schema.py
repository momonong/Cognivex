# scripts/graph_rag/get_graph_schema.py
from .neo_client import run_cypher_query
from collections import defaultdict
import json
from agents.llm_client.gemini_client import gemini_chat




def get_graph_schema() -> dict:
    schema = {
        "nodes": defaultdict(list),  # label -> list of properties
        "relationships": defaultdict(list),  # rel type -> list of properties
    }

    # 1. Node types
    node_query = """
    CALL db.schema.nodeTypeProperties()
    YIELD nodeLabels, propertyName, propertyTypes
    RETURN nodeLabels, propertyName, propertyTypes
    """
    nodes = run_cypher_query(node_query)
    for row in nodes:
        for label in row["nodeLabels"]:
            schema["nodes"][label].append(
                {"property": row["propertyName"], "type": row["propertyTypes"]}
            )

    # 2. Relationship types
    rel_query = """
    CALL db.schema.relTypeProperties()
    YIELD relType, propertyName, propertyTypes
    RETURN relType, propertyName, propertyTypes
    """
    rels = run_cypher_query(rel_query)
    for row in rels:
        schema["relationships"][row["relType"]].append(
            {"property": row["propertyName"], "type": row["propertyTypes"]}
        )

    return schema


def describe_schema_with_llm(schema: dict) -> str:
    """使用 Gemini 對 graph schema 進行自然語言摘要"""
    prompt = f"""
        You are a knowledge graph assistant. Your task is to read a graph database schema and describe its structure in natural language.

        Schema format is JSON with the following fields:
        - nodes: Each node type with its properties and data types.
        - relationships: Each relationship type with their property keys and types.

        Explain:
        1. What kinds of entities exist (node types)?
        2. What relationships exist between them?
        3. What properties each node/edge type has?
        4. Provide a high-level summary of what this graph seems to describe.

        Be concise, but accurate and informative.

        Schema:
        {json.dumps(schema, indent=2)}
    """
    return gemini_chat(prompt=prompt, mime_type="text/plain")


if __name__ == "__main__":
    schema = get_graph_schema()
    print("JSON Schema:")
    print(json.dumps(schema, indent=2, ensure_ascii=False))

    print("\nNatural Language Summary from Gemini:")
    summary = describe_schema_with_llm(schema)
    print(summary)
