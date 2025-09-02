from typing import List, Tuple, Dict
import json # 為了回傳給 Agent，我們最後會轉成 JSON 字串

def evaluate_query(
    result_json: str, # <-- 輸入改為 JSON 字串，與其他工具統一
    question: str,
    expected_fields: List[str],
    min_count: int,
) -> str: # <-- 輸出也改為 JSON 字串，方便 Agent 解析
    """
    Decide if query results are sufficient, and return evaluation summary as a JSON string.
    This robust version handles case-insensitivity for field names.

    Returns:
        A JSON string containing the evaluation result:
        {
            "should_retry": bool,
            "is_valid": bool,
            "reason": str,
            "suggested_strategy": str | None
        }
    """
    print(f"\nEvaluating query result for retry decision...")
    print(f"- Question: {question}")
    print(f"- Expected Fields: {expected_fields}")
    print(f"- Min Count: {min_count}")

    eval_result = {}
    
    try:
        results = json.loads(result_json)
        if not isinstance(results, list):
            raise TypeError("Result is not a list.")
    except (json.JSONDecodeError, TypeError) as e:
        eval_result = {
            "should_retry": True,
            "is_valid": False,
            "reason": f"Result is not a valid JSON list of objects. Error: {e}",
            "suggested_strategy": "The query result was malformed. Please re-run the query and ensure the output is a valid JSON list."
        }
        print(f"function call: evaluate_query() FAILED - {eval_result['reason']}")
        return json.dumps(eval_result)

    print(f"- Result Count: {len(results)}")

    # --- 1. 檢查結果數量是否達標 ---
    if len(results) < min_count:
        eval_result = {
            "should_retry": True,
            "is_valid": False,
            "reason": f"Too few results returned: expected >= {min_count}, got {len(results)}",
            "suggested_strategy": "Consider relaxing query filters or using broader search terms."
        }
        print(f"function call: evaluate_query() FAILED - {eval_result['reason']}")
        return json.dumps(eval_result)

    # --- 2. 穩健的欄位檢查 (只檢查第一筆結果，假設 Schema 一致) ---
    if results: # 確保 results 不是空的
        first_result_keys = {key.lower() for key in results[0].keys()} # <-- 將實際 key 轉為小寫
        expected_fields_lower = {field.lower() for field in expected_fields} # <-- 將預期 key 轉為小寫

        if not expected_fields_lower.issubset(first_result_keys):
            missing_fields = [f for f in expected_fields if f.lower() not in first_result_keys]
            eval_result = {
                "should_retry": True,
                "is_valid": False,
                "reason": f"Missing expected fields in results: {missing_fields}. Found: {list(results[0].keys())}",
                "suggested_strategy": (
                    f"The query result is missing critical fields: {missing_fields}. "
                    f"Please regenerate the Cypher query. Ensure the RETURN clause includes these fields, possibly using an alias like 'AS {missing_fields[0]}'."
                )
            }
            print(f"function call: evaluate_query() FAILED - {eval_result['reason']}")
            return json.dumps(eval_result)

    # --- 3. 所有檢查都通過 ---
    print("function call: evaluate_query() PASSED")
    eval_result = {
        "should_retry": False,
        "is_valid": True,
        "reason": "Query result is sufficient and has all required fields.",
        "suggested_strategy": None
    }
    return json.dumps(eval_result)