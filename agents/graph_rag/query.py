# agents/graph_rag/query.py
from agents.graph_rag.schema import get_graph_schema
from agents.graph_rag.client import gemini_chat
import json

def build_cypher_prompt(question: str, schema: dict) -> str:
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    
    prompt = f"""
        You are a Cypher query expert for a Neo4j knowledge graph. 
        The user will provide a natural language question. Based on the schema below, you will generate an appropriate Cypher query.

        == Graph Schema ==
        {schema_text}

        == Task ==
        Write a Cypher query that answers the following question:
        "{question}"

        Ensure the Cypher query is valid and matches the schema. Do not add explanation or markdown.

        == Cypher ==
    """
    return prompt.strip()

def generate_cypher(question: str) -> str:
    schema = get_graph_schema()
    prompt = build_cypher_prompt(question, schema)
    
    print("📤 Prompt sent to Gemini:")
    print(prompt[:500] + "...\n")  # 只印前 500 字

    response = gemini_chat(prompt)
    return response.strip()


if __name__ == "__main__":
    question = input("🔍 請輸入問題: ")
    cypher = generate_cypher(question)
    print("\n🧠 Cypher generated:\n", cypher)
