import json
from typing import List
from pydantic import BaseModel, Field # 匯入 Pydantic

from app.services.llm_providers import llm_response # 假設您使用 router
from app.services.neo4j_connector import get_neo4j_driver

driver = get_neo4j_driver()

class CanonicalNameList(BaseModel):
    """用於保存匹配後的大腦標準區域名稱列表的資料模型。"""
    matched_names: List[str] = Field(description="一個只包含匹配成功的大腦標準區域名稱的列表。")


def get_all_region_names() -> List[str]:
    query = "MATCH (r:Region) RETURN r.name AS name ORDER BY r.name"
    with driver.session() as session:
        result = session.run(query)
        region_names = [record["name"] for record in result]
    return region_names


def entity_linker_tool(dirty_region_list: list[str]) -> list[str]:
    """
    使用 LLM 將一個 'dirty' 列表與資料庫中的權威名稱列表進行匹配，
    並回傳一個保證存在於資料庫中的 'clean' 列表。
    這個版本使用了 response_schema 來確保輸出的可靠性。
    """

    canonical_regions = get_all_region_names()
    if not canonical_regions:
        return []

    # 2. 【建議修改】稍微調整 Prompt，使其更明確地引導模型填充 Pydantic 模型
    prompt = f"""
    You are a data-matching assistant. Your task is to align the 'dirty list' of brain regions
    with the 'canonical list' and return a JSON object containing only the matched canonical names.
    
    CRITICAL MATCHING RULE: The canonical names have a format like "Region_Hemisphere Number"
    (e.g., "Angular_R 70"). For a name like "Angular_R" in the dirty list, you MUST find the entry
    in the canonical list that STARTS WITH "Angular_R". Return ONLY the full canonical name from the list.

    Your final output MUST be a JSON object with a single key "matched_names" which contains a list of the matched strings.

    ---
    **Dirty List:** {json.dumps(dirty_region_list)}
    ---
    **Canonical List (All valid options):** {json.dumps(canonical_regions)}
    ---
    """

    try:
        # 3. 【核心修改】在呼叫 LLM 時，傳遞我們定義好的 Pydantic 模型
        response_str = llm_response(
            prompt,
            response_schema=CanonicalNameList # <--- 關鍵！
        )
        
        # 4. 【核心修改】解析現在絕對乾淨的 JSON 輸出
        data = json.loads(response_str)
        clean_list = data.get('matched_names', []) # 從包裹中安全地取出真正的列表
        return clean_list
    except Exception as e:
        print(f"[Tool Log - ERROR] Failed during entity linking: {e}")
        # 這裡的錯誤現在很可能是真正的網路問題或 API 錯誤，而不是 JSON 解析錯誤
        return []