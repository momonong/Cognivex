# app/agents/report_generation.py
from typing import List, Dict, Any
from app.graph.state import AgentState, BrainRegionInfo
from app.services.llm_provider import llm_response

# 輔助函數 format_regions_for_prompt 幾乎可以保持不變，或稍微簡化
def format_regions_for_prompt(regions: List[BrainRegionInfo]) -> str:
    if not regions:
        return "No significant brain region activations were identified."
    text_parts = []
    for region in regions[:5]: # 我們只需要最重要的幾個腦區來做摘要
        name = region.get("region_name", "N/A")
        networks = ", ".join(region.get("associated_networks", [])) or "N/A"
        functions = ", ".join(region.get("known_functions", [])) or "N/A"
        text_parts.append(f"- {name} (Belongs to: {networks}; Known for: {functions})")
    return "\n".join(text_parts)

def generate_final_report(state: AgentState) -> dict:
    """
    Node: Synthesizes information into a structured JSON report for the dashboard.
    This version uses focused LLM calls for specific fields.
    """
    print("\n--- Node: Structured Report Generator ---")

    try:
        # 1. 收集所有需要的資訊
        classification = state.get("classification_result", "N/A")
        enriched_regions = state.get("activated_regions", [])
        subject_id = state.get("subject_id", "N/A")
        visualization_paths = state.get("visualization_paths", {})
        
        # =================================================================
        # 2. 模組化生成報告內容
        # =================================================================

        # --- Part A: 直接從 state 轉換的資料 (無需 LLM) ---
        print("  - Formatting direct data fields...")
        activation_analysis = {
            "title": "顯著腦區活動分析",
            "table_headers": ["腦區名稱", "所屬功能網絡", "關聯之臨床功能", "激活強度"],
            "regions": [
                {
                    "name": r.get("region_name", "N/A"),
                    "network": ", ".join(r.get("associated_networks", [])),
                    "function": ", ".join(r.get("known_functions", [])),
                    "activation": r.get("activation_score", 0)
                } for r in enriched_regions
            ]
        }

        # --- Part B: 需要 LLM 進行摘要和推理的欄位 ---
        formatted_regions_text = format_regions_for_prompt(enriched_regions)
        
        # LLM 任務 1: 生成診斷摘要 (Diagnostic Summary)
        print("  - Generating diagnostic summary with LLM...")
        summary_prompt = f"""
        Given the following fMRI analysis data for subject {subject_id}:
        - Initial Classification: {classification}
        - Key Activated Regions: {formatted_regions_text}
        
        Task: Write a single, concise sentence for a clinical report's "Key Finding" section. This sentence should synthesize the classification with the primary brain network involved.
        Example: "腦部活動模式與預設模式網絡 (DMN) 的功能異常高度相關，此為阿茲哈默症的典型神經病理特徵。"
        """
        key_finding_en = llm_response(prompt=summary_prompt, llm_provider="gemini")

        # LLM 任務 2: 生成臨床推理 (Clinical Reasoning)
        print("  - Generating clinical reasoning with LLM...")
        reasoning_prompt = f"""
        You are a neuroradiologist AI. Based on the data below, write a brief, professional "Clinical Reasoning" narrative (2-4 sentences).
        - Initial Classification: {classification}
        - Key Activated Regions: {formatted_regions_text}
        
        Task: Explain WHY these findings support the classification, linking the brain networks (e.g., DMN) to the disease's known symptoms (e.g., memory loss).
        """
        reasoning_narrative_en = llm_response(prompt=reasoning_prompt, llm_provider="gemini")
        
        # --- Part C: 翻譯生成的文字欄位 ---
        print("  - Translating generated text fields to Chinese...")
        key_finding_zh = llm_response(prompt=f"Translate this clinical finding to Traditional Chinese: '{key_finding_en}'", llm_provider="gemini")
        reasoning_narrative_zh = llm_response(prompt=f"Translate this clinical narrative to Traditional Chinese: '{reasoning_narrative_en}'", llm_provider="gemini")

        # =================================================================
        # 3. 組裝成最終的 JSON 結構
        # =================================================================
        print("  - Assembling final JSON report...")
        final_report_json = {
            "subject_info": {"subject_id": subject_id},
            "diagnostic_summary": {
                "prediction": "阿茲海默症 (AD)" if classification == "AD" else "認知正常 (CN)",
                "confidence_score": state.get("confidence_score", 0.0), # 假設 state 中有這個值
                "key_finding": {"en": key_finding_en.strip(), "zh": key_finding_zh.strip()}
            },
            "activation_analysis": activation_analysis,
            "clinical_reasoning": {
                "title": "臨床推理說明",
                "narrative": {"en": reasoning_narrative_en.strip(), "zh": reasoning_narrative_zh.strip()}
            },
            "visualization_paths": visualization_paths
        }
        
        trace = "Node: Structured report generation complete."
        return {
            "final_report_json": final_report_json,
            "trace_log": state.get("trace_log", []) + [trace]
        }
        
    except Exception as e:
        error_message = f"Node (Structured Report Generator) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}