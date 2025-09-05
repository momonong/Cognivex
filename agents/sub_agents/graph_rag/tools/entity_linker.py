# in tools/linker_tool.py
import json
from typing import List

from agents.client.llm_client import llm_response
from agents.client.neo4j_client import get_neo4j_driver

driver = get_neo4j_driver()


def get_all_region_names() -> List[str]:
    """
    Get all brain region names from the Neo4j database.
    """
    query = "MATCH (r:Region) RETURN r.name AS name ORDER BY r.name"
    with driver.session() as session:
        result = session.run(query)
        region_names = [record["name"] for record in result]
    return region_names


def entity_linker_tool(dirty_region_list: list[str]) -> list[str]:
    """
    Takes a 'dirty' list of region names and matches them against a canonical
    list from the database using an LLM to produce a 'clean' list of names
    that are guaranteed to exist in the database.
    """

    # 1. 獲取權威名單
    canonical_regions = get_all_region_names()
    if not canonical_regions:
        return []

    # 2. 設計 Prompt，讓 LLM 進行模糊匹配
    prompt = f"""
    You are a data-matching assistant. Align the 'dirty list' of brain regions
    with the 'canonical list' from the database.
    
    CRITICAL MATCHING RULE: The canonical names have a format like "Region_Hemisphere Number"
    (e.g., "Angular_R 70"). For a name like "Angular_R" in the dirty list, you MUST find the entry
    in the canonical list that STARTS WITH "Angular_R". Return ONLY the full canonical name.

    Your final output MUST be a single, clean JSON list of the matched canonical names.

    ---
    **Dirty List:** {json.dumps(dirty_region_list)}
    ---
    **Canonical List (All valid options):** {json.dumps(canonical_regions)}
    ---
    """

    print("--- [Tool Log] Running LLM for entity linking... ---")
    # 3. 呼叫 LLM 並解析結果
    try:
        response_str = llm_response(prompt)
        clean_list = json.loads(
            response_str.strip().replace("```json", "").replace("```", "")
        )
        print(f"--- [Tool Log] Entity Linking complete. Clean list: {clean_list} ---")
        return clean_list
    except Exception as e:
        print(f"--- [Tool Log - ERROR] Failed during entity linking: {e} ---")
        return []
