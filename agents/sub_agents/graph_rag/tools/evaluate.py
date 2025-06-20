from typing import List, Tuple, Dict


def evaluate_query(
    results: List[dict],
    question: str,
    expected_fields: List[str],
    min_count: int = 1,
) -> Tuple[bool, Dict]:
    """
    Decide if query results are sufficient, and return evaluation summary.

    Returns:
        should_retry (bool): whether result is insufficient
        eval_dict (dict): fields = is_valid, reason, suggested_strategy
    """

    print(f"\nEvaluating query result for retry decision...")
    print(f"- Question: {question}")
    print(f"- Expected Fields: {expected_fields}")
    print(f"- Result Count: {len(results)}")

    # --- 1. Empty result ---
    if not results:
        return True, {
            "is_valid": False,
            "reason": "No results returned.",
            "suggested_strategy": (
                "Try expanding the query with broader relationships or removing overly strict filters."
            ),
        }

    # --- 2. Check expected fields existence in result rows ---
    found_fields = set()
    for row in results:
        for f in expected_fields:
            if f in row and isinstance(row[f], str) and row[f].strip():
                found_fields.add(f)

    missing_fields = [f for f in expected_fields if f not in found_fields]
    if missing_fields:
        return True, {
            "is_valid": False,
            "reason": f"Missing expected fields in results: {missing_fields}",
            "suggested_strategy": (
                "Ensure these fields are in RETURN clause or use OPTIONAL MATCH to fetch them."
            ),
        }

    # --- 3. Check result count threshold ---
    if len(results) < min_count:
        return True, {
            "is_valid": False,
            "reason": f"Too few results returned: expected ≥ {min_count}, got {len(results)}",
            "suggested_strategy": "Consider relaxing filters or adding fallbacks in query logic.",
        }

    # --- 4. All checks passed ---
    print("function call: should_retry_query() PASSED")
    return False, {
        "is_valid": True,
        "reason": "Query result is sufficient.",
        "suggested_strategy": None,
    }
