import json
import re

def parse_gemini_json_response(text: str):
    """
    Extract and parse JSON object from Gemini response, even if wrapped in ```json ... ``` fences.
    """
    # 嘗試從 markdown code fence 中取出 JSON
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
