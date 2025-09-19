# app/agents/entity_linking.py
from typing import List
from app.graph.state import AgentState, BrainRegionInfo
# 從 core 工具庫中 import 我們剛剛遷移的工具
from app.core.knowledge_graph.entity_linker import entity_linker_tool

def link_entities(state: AgentState) -> dict:
    """
    Node: Performs entity linking on brain region names.
    Takes the potentially 'dirty' list of names from post-processing,
    and uses the entity_linker_tool to produce a clean list.
    """
    print("\n--- Node: Entity Linking ---")
    activated_regions: List[BrainRegionInfo] = state.get("activated_regions")

    if not activated_regions:
        print("  - No regions to link. Skipping.")
        return {"clean_region_names": []}

    # 1. 準備 'dirty' list
    dirty_region_names = [region["region_name"] for region in activated_regions]
    print(f"  - Linking {len(dirty_region_names)} region names...")
    
    # 2. 調用 entity_linker_tool
    clean_names = entity_linker_tool(dirty_region_names)
    print(f"  - Linking complete. Found {len(clean_names)} valid names.")

    trace = f"Node: Entity Linking - Cleaned {len(dirty_region_names)} names down to {len(clean_names)}."

    return {
        "clean_region_names": clean_names, # 將 clean list 存入 state
        "trace_log": state.get("trace_log", []) + [trace]
    }