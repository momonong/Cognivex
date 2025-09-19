# app/agents/knowledge_reasoning.py
import json
from typing import List
from app.graph.state import AgentState, BrainRegionInfo
# 從我們的 core 工具庫中 import graphrag 工具
from app.core.knowledge_graph.query_engine import graphrag

def enrich_with_knowledge_graph(state: AgentState) -> dict:
    """
    Node: Enriches brain regions using the CLEAN list of names from the previous step.

    This node takes the cleaned list of region names, queries the knowledge graph
    for associated networks and functions, and merges this new information back
    into the main 'activated_regions' list in the state.
    """
    print("\n--- Node: Knowledge Graph Enrichment ---")

    # 1. 從 state 中獲取所需要的數據
    clean_region_names: List[str] = state.get("clean_region_names")
    original_regions: List[BrainRegionInfo] = state.get("activated_regions")

    # 2. 條件檢查：如果沒有數據需要處理，則提前退出
    if not clean_region_names or not original_regions:
        print("  - No clean region names or original regions found to enrich. Skipping.")
        return {}

    print(f"  - Querying KG for {len(clean_region_names)} clean region names...")

    # 3. 調用 graphrag 工具
    json_result_str = graphrag(clean_region_names)
    
    # 4. 解析工具回傳的 JSON 字串
    try:
        kg_data = json.loads(json_result_str)
    except json.JSONDecodeError as e:
        error_message = f"Node (Knowledge Reasoning) Error: Failed to parse JSON response from graphrag tool. Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}

    # 5. 處理工具本身可能回報的錯誤
    if "error" in kg_data:
        error_message = f"Node (Knowledge Reasoning) Error: {kg_data['error']}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}

    kg_results = kg_data.get("result", [])
    
    # 6. 為了方便快速查找，將 KG 結果轉換成一個字典 (lookup map)
    #    鍵是腦區名稱，值是從 KG 查回的完整記錄
    results_map = {item["region"]: item for item in kg_results}
    print(f"  - Successfully retrieved KG data for {len(results_map)} regions.")

    # 7. 將 KG 的結果合併回我們原有的 activated_regions 列表中
    enriched_regions = []
    for region_info in original_regions:
        # 建議在修改前先複製一份，是良好的程式撰寫習慣
        enriched_region_info = region_info.copy()
        region_name = enriched_region_info.get("region_name")
        
        # 檢查這個腦區是否有對應的 KG 數據
        if region_name in results_map:
            kg_info = results_map[region_name]
            
            # 安全地獲取 network 和 functions 資訊
            network = kg_info.get("network")
            functions = kg_info.get("functions", [])
            
            # 更新字典，加入新的鍵值對
            enriched_region_info.update({
                "associated_networks": [network] if network else [],
                "known_functions": functions
            })
        
        enriched_regions.append(enriched_region_info)

    trace = f"Node: Knowledge graph enrichment complete. Enriched {len(results_map)} regions."
    
    # 8. 回傳更新後的列表，它將覆蓋掉 state 中舊的 activated_regions
    return {
        "activated_regions": enriched_regions,
        "trace_log": state.get("trace_log", []) + [trace]
    }