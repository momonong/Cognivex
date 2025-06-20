# agents/sub_agents/graph_rag/tools/evaluate.py
import json
from typing import List, Dict, Optional
from agents.llm_client.gemini_client import gemini_chat


def rule_based_evaluate(
    results: List[Dict], expected_fields: Optional[List[str]] = None, min_count: int = 1
) -> Optional[Dict]:
    """
    Perform simple rule-based checks on query results.
    Returns a dict with evaluation info if rules fail, or None if rules pass.

    - No results returns is_valid=False with strategy 'relax'.
    - Fewer than min_count rows returns is_valid=False with strategy 'optional'.
    - Missing expected field returns is_valid=False with strategy 'drop_filters'.
    """
    if not results:
        return {
            "is_valid": False,
            "reason": "No results returned",
            "suggested_strategy": "relax",
        }
    if len(results) < min_count:
        return {
            "is_valid": False,
            "reason": f"Expected at least {min_count} rows but got {len(results)}",
            "suggested_strategy": "optional",
        }
    if expected_fields:
        for row in results[:min_count]:
            for field in expected_fields:
                if field not in row:
                    return {
                        "is_valid": False,
                        "reason": f"Missing expected field '{field}'",
                        "suggested_strategy": "drop_filters",
                    }
    return None


def llm_based_evaluate(
    results: List[Dict], question: str, sample_size: int = 3
) -> Dict:
    """
    Use an LLM to semantically evaluate whether results answer the question.
    Returns a dict with keys: is_valid, reason, suggested_strategy.
    """
    snippet = results[:sample_size]
    prompt = f"""
        You are an evaluator for graph query results.

        User Question:
        {question}

        Result Snippet:
        {json.dumps(snippet, indent=2)}

        Instructions:
        - Determine if this snippet fully answers the question.
        - Respond with a JSON object:
        {{"is_valid": bool, "reason": str, "suggested_strategy": str or null}}
        - Do not output any additional text.
    """
    raw = gemini_chat(prompt=prompt, mime_type="application/json")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "is_valid": False,
            "reason": "Invalid evaluation response format",
            "suggested_strategy": "relax",
        }


def evaluate_and_decide(
    results: List[Dict],
    question: str,
    expected_fields: Optional[List[str]] = None,
    min_count: int = 1,
) -> Dict:
    """
    Combine rule-based and LLM-based evaluation to decide query quality.
    Returns a dict with is_valid, reason, suggested_strategy.
    """
    # First apply rule-based evaluation
    rule_eval = rule_based_evaluate(results, expected_fields, min_count)
    if rule_eval is not None:
        return rule_eval

    # Fall back to LLM-based evaluation
    return llm_based_evaluate(results, question)
