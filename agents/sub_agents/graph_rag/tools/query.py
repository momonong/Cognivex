import json
from typing import Optional, List
from pydantic import BaseModel

from agents.llm_client.gemini_client import gemini_chat
from .neo4j_client import run_cypher_query
from .schema import summarize_graph_schema


# -----------------------
# Output Schema for GraphRAG
# -----------------------


class QueryEvaluation(BaseModel):
    is_valid: bool
    reason: Optional[str] = None
    suggested_strategy: Optional[str] = None


class GraphRAGOutput(BaseModel):
    answer: str
    cypher: str
    regions: List[str] = []
    functions: List[str] = []
    evaluation: QueryEvaluation = QueryEvaluation(
        is_valid=True, reason="Not yet evaluated", suggested_strategy=None
    )


# -----------------------
# Cypher Prompt Templates
# -----------------------

TEMPLATE_MAP = {
    "ad_region_functions": lambda _: (
        """
        MATCH (r:Region {ad_associated: true})
        MATCH (r)-[:HAS_FUNCTION]->(f:Function)
        RETURN DISTINCT r.name AS region, f.name AS function
        """
    ),
    "region_info_by_keyword": lambda kw: f"""
        MATCH (r:Region)
        WHERE toLower(r.name) CONTAINS toLower('{kw}')
        OPTIONAL MATCH (r)-[:HAS_FUNCTION]->(f:Function)
        RETURN r.name AS region, r.ad_associated AS ad_flag, collect(f.name) AS functions
    """,
}


def classify_question_type(question: str):
    q = question.lower()
    if "alzheimer" in q and "function" in q:
        return "ad_region_functions"
    if "region" in q and "info" in q:
        return "region_info_by_keyword"
    return None


# -----------------------
# 1. Cypher Generation
# -----------------------


def generate_cypher(question: str, schema_text: str) -> str:
    qtype = classify_question_type(question)
    if qtype and qtype in TEMPLATE_MAP:
        query = TEMPLATE_MAP[qtype](question).strip()
        print(f"Using template '{qtype}' for query generation.")
        return query

    print("\n🧠 Generating Cypher via LLM...")
    base_prompt = f"""
        You are a Cypher expert. Given the graph schema and a user question, generate a MATCH query.
        - Use MATCH/OPTIONAL MATCH on existing nodes and relationships.
        - Use case-insensitive matching.
        - Return only the Cypher query, no markdown.
        If possible, include both region names and their functions in your answer.
    """
    return generate_cypher_from_prompt(question, schema_text, base_prompt)


def generate_cypher_from_prompt(
    question: str,
    schema_text: str,
    base_prompt: str,
    extra_context: Optional[str] = None,
) -> str:
    prompt = f"""
        {base_prompt.strip()}

        Graph Schema:
        {schema_text.strip()}

        User Question:
        {question.strip()}
    """
    if extra_context:
        prompt += f"\n\n{extra_context.strip()}"

    prompt += "\n\nCypher Query:"
    raw = gemini_chat(prompt=prompt, mime_type="text/plain")
    return raw.replace("```cypher", "").replace("```", "").strip()


# -----------------------
# 2. Cypher Update（with strategy）
# -----------------------


def regenerate_cypher_with_strategy(
    original_cypher: str,
    question: str,
    strategy: str,
    schema_text: str,
) -> str:
    print(f"\nRewriting Cypher with strategy: {strategy}")
    base_prompt = f"""
        The following Cypher query returned poor or incomplete results:
        {original_cypher}
    """
    extra = f"Strategy: {strategy}\nPlease rewrite the query using this strategy. Return only the new Cypher."
    return generate_cypher_from_prompt(question, schema_text, base_prompt, extra)


# -----------------------
# 3. Cypher Execution
# -----------------------


def run_cypher(cypher: str) -> list:
    print("\nExecuting Cypher query:")
    print(cypher)
    return run_cypher_query(cypher)


# -----------------------
# 4. Final Answer Synthesis
# -----------------------


def synthesize_answer(question: str, result: list) -> str:
    prompt = f"""
        You are a helpful assistant. Answer the question concisely based on the data below.

        Question:
        {question}

        Data:
        {json.dumps(result[:10], indent=2)}

        Answer:
    """
    return gemini_chat(prompt=prompt, mime_type="text/plain").strip()


# -----------------------
# 5. GraphRAG External Interface（Single Run）
# -----------------------


def graph_rag_query(question: str) -> GraphRAGOutput:
    """
    Returns answer, cypher used, and raw region/function results.
    Evaluation is deferred to the agent.
    """
    schema_text = summarize_graph_schema()
    cypher = generate_cypher(question, schema_text)
    result = run_cypher(cypher)
    answer = synthesize_answer(question, result)

    regions = [r.get("region") for r in result if isinstance(r.get("region"), str)]
    functions = [
        r.get("function") for r in result if isinstance(r.get("function"), str)
    ]

    print("function call: graph_rag_query()")
    return GraphRAGOutput(
        answer=answer,
        cypher=cypher,
        regions=regions,
        functions=functions,
        evaluation=QueryEvaluation(
            is_valid=True,
            reason="Evaluation deferred to agent.",
            suggested_strategy=None
        )
    ).model_dump()



# -----------------------
# CLI Test
# -----------------------

if __name__ == "__main__":
    q = "What functions are performed by brain regions associated with Alzheimer's disease?"
    out = graph_rag_query(q)
    print("\nResult Object:\n", out.json(indent=2))
