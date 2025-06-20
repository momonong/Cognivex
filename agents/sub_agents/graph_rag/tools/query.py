# agents/sub_agents/graph_rag/tools/query.py
import json
from typing import Optional
from agents.llm_client.gemini_client import gemini_chat
from .neo4j_client import run_cypher_query
from .schema import summarize_graph_schema
from .evaluate import evaluate_and_decide

# -----------------------
# 1. Define Query Templates
# -----------------------
TEMPLATE_MAP = {
    "ad_region_functions": lambda _: (
        """
        MATCH (r:Region {ad_associated: true})
        MATCH (r)-[:HAS_FUNCTION]->(f:Function)
        RETURN DISTINCT f.name AS function_name
        """
    ),
    "region_info_by_keyword": lambda kw: f"""
        MATCH (r:Region)
        WHERE toLower(r.name) CONTAINS toLower('{kw}')
        OPTIONAL MATCH (r)-[:HAS_FUNCTION]->(f:Function)
        RETURN r.name AS region, r.ad_associated AS ad_flag, collect(f.name) AS functions
    """
}

# -----------------------
# 2. Classify Question
# -----------------------

def classify_question_type(question: str) -> Optional[str]:
    q = question.lower()
    if "alzheimer" in q and "function" in q:
        return "ad_region_functions"
    if "region" in q and "info" in q:
        return "region_info_by_keyword"
    return None

# -----------------------
# 3. Generate Cypher
# -----------------------

def generate_cypher(question: str, schema_text: str) -> str:
    qtype = classify_question_type(question)
    if qtype and qtype in TEMPLATE_MAP:
        query = TEMPLATE_MAP[qtype](question).strip()
        print(f"Using template '{qtype}' for query generation.")
        return query

    prompt = f"""
        You are a Cypher expert. Given the graph schema below and a user question, generate a MATCH query.

        Graph Schema:
        {schema_text}

        User Question:
        {question}

        Instructions:
        - Use MATCH/OPTIONAL MATCH on existing nodes and relationships.
        - Use case-insensitive matching where appropriate.
        - Return only the Cypher query, no markdown.

        Cypher Query:
    """
    print("\nGenerating Cypher via LLM...")
    raw = gemini_chat(prompt=prompt, mime_type="text/plain")
    query = raw.replace('```cypher', '').replace('```', '').strip()
    print(f"Generated Cypher:\n{query}")
    return query

# -----------------------
# 4. Execute Query
# -----------------------

def run_query(cypher: str) -> list:
    print("\n📡 Executing Cypher query:")
    print(cypher)
    return run_cypher_query(cypher)

# -----------------------
# 5. Adjust Query (update)
# -----------------------

def adjust_cypher_query(original: str, question: str, strategy: str, schema_text: str) -> str:
    print(f"\n🛠 Evaluation failed, applying strategy: {strategy}")
    prompt = f"""
        The following Cypher query returned poor or no results:
        {original}

        Strategy: {strategy}

        Graph Schema:
        {schema_text}

        User Question:
        {question}

        Please rewrite the query using this strategy. Return only the new Cypher.
    """
    raw = gemini_chat(prompt=prompt, mime_type="text/plain")
    new_query = raw.replace('```cypher', '').replace('```', '').strip()
    print(f"Adjusted Cypher:\n{new_query}")
    return new_query

# -----------------------
# 6. Orchestrator with Update Logic
# -----------------------

def graph_rag_answer(question: str) -> str:
    # 1. Extract schema
    schema_text = summarize_graph_schema()
    print("\nRetrieved schema and formatted for LLM.")

    # 2. Generate initial query
    cypher = generate_cypher(question, schema_text)

    # 3. Execute initial query
    result = run_query(cypher)

    # 4. Evaluate results
    eval_info = evaluate_and_decide(
        results=result,
        question=question,
        expected_fields=["function_name"],
        min_count=1
    )
    print(f"\nEvaluation result: {eval_info}")

    # 5. If evaluation failed, adjust query
    if not eval_info.get("is_valid", False):
        cypher = adjust_cypher_query(cypher, question, eval_info.get("suggested_strategy"), schema_text)
        result = run_query(cypher)

    # 6. Synthesize final answer
    print("\nSynthesizing final answer...")
    prompt = f"""
        You are a helpful assistant. Answer the question concisely based on data below.

        Question:
        {question}

        Data:
        {json.dumps(result[:10], indent=2)}

        Answer:
    """
    answer = gemini_chat(prompt=prompt, mime_type="text/plain").strip()
    return answer

# -----------------------
# CLI Testing
# -----------------------

if __name__ == '__main__':
    user_q = "What functions are performed by brain regions associated with Alzheimer's disease?"
    final_answer = graph_rag_answer(user_q)
    print("\nFinal Answer:\n", final_answer)
