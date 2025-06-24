from typing import List, Optional
from pydantic import BaseModel


class ResultRow(BaseModel):
    field1: Optional[str] = None
    field2: Optional[str] = None


def evaluate_query(
    results: List[ResultRow],
    question: str,
    expected_fields: List[str],
    min_count: Optional[int] = 1,
) -> dict:
    print(f"\nEvaluating query result for retry decision...")
    print(f"- Question: {question}")
    print(f"- Expected Fields: {expected_fields}")
    print(f"- Result Count: {len(results)}")

    # Prepare final message
    def format_result(reason: str, suggestion: Optional[str] = None):
        msg = f"Query analysis: {reason}"
        if suggestion:
            msg += f" Suggested fix: {suggestion}"
        return msg

    if not results:
        reason = "No results returned."
        suggestion = "Try expanding the query with broader relationships or removing overly strict filters."
        return {
            "graph_rag_result": format_result(reason, suggestion),
            "should_retry": True,
            "is_valid": False,
            "reason": reason,
            "suggested_strategy": suggestion,
        }

    found_fields = set()
    for row in results:
        for f in expected_fields:
            value = getattr(row, f, None)
            if isinstance(value, str) and value.strip():
                found_fields.add(f)

    missing_fields = [f for f in expected_fields if f not in found_fields]
    if missing_fields:
        reason = f"Missing expected fields in results: {missing_fields}"
        suggestion = "Ensure these fields are in RETURN clause or use OPTIONAL MATCH to fetch them."
        return {
            "graph_rag_result": format_result(reason, suggestion),
            "should_retry": True,
            "is_valid": False,
            "reason": reason,
            "suggested_strategy": suggestion,
        }

    if len(results) < min_count:
        reason = f"Too few results returned: expected ≥ {min_count}, got {len(results)}"
        suggestion = "Consider relaxing filters or adding fallbacks in query logic."
        return {
            "graph_rag_result": format_result(reason, suggestion),
            "should_retry": True,
            "is_valid": False,
            "reason": reason,
            "suggested_strategy": suggestion,
        }

    # All checks passed
    reason = "Query result is sufficient."
    print("function call: should_retry_query() PASSED")
    return {
        "graph_rag_result": format_result(reason),
        "should_retry": False,
        "is_valid": True,
        "reason": reason,
        "suggested_strategy": None,
    }
