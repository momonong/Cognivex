# app/agents/report_generation.py
from typing import List
from app.graph.state import AgentState, BrainRegionInfo
# 導入我們真實的 LLM 服務
from app.services.llm_provider import llm_response


def format_regions_for_prompt(regions: List[BrainRegionInfo]) -> str:
    # ... (這個輔助函數保持不變)
    if not regions:
        return "No significant brain region activations were identified."

    text_parts = ["Key Activated Regions and Their Known Associations:\n"]
    for region in regions[:15]: 
        name = region.get("region_name", "N/A")
        score = region.get("activation_score", 0)
        networks = ", ".join(region.get("associated_networks", [])) or "N/A"
        functions = ", ".join(region.get("known_functions", [])) or "N/A"
        
        text_parts.append(
            f"- **{name}** (Activation Score: {score:.3f})\n"
            f"  - Associated Networks: {networks}\n"
            f"  - Known Functions: {functions}\n"
        )
    return "\n".join(text_parts)

def generate_final_report(state: AgentState) -> dict:
    """
    Node: Synthesizes all information from the state into a final,
    structured report using a real LLM call.
    """
    print("\n--- Node: Final Report Generator ---")
    
    # 1. 收集所有資訊 (保持不變)
    classification = state.get("classification_result")
    enriched_regions = state.get("activated_regions")
    image_explanation_obj = state.get("image_explanation", {})
    image_explanation_text = image_explanation_obj.get("text", "No visual analysis available.")
    subject_id = state.get("subject_id")
    
    # 2. 建立 Prompt (保持不變)
    formatted_regions_text = format_regions_for_prompt(enriched_regions)
    synthesis_prompt = f"""
    You are a professional neuroradiologist AI. Your task is to generate a comprehensive clinical fMRI report for subject: {subject_id}.
    Synthesize all the following information into a single, fluent, and well-structured report.

    **--- Primary Finding ---**
    The initial deep learning model classification for this subject is: **{classification}**

    **--- Visual Analysis of Activation Maps ---**
    An expert vision model provided the following summary of the fMRI activation maps:
    "{image_explanation_text}"

    **--- Detailed Brain Region Analysis (Data from KG) ---**
    {formatted_regions_text}

    **--- YOUR TASK ---**
    Based on ALL the information above, write the final clinical summary report. The report must be structured with the following sections EXACTLY:
    - **Primary Assessment Finding**
    - **Interpretation of Brain Activity Patterns**
    - **Correlation with Established Neurological Knowledge**
    - **Conclusion**
    """

    print("  - Sending final synthesis prompt to LLM...")
    try:
        # --- REPLACE MOCK DATA WITH REAL LLM CALLS ---
        
        # Call for English report
        final_report_en = llm_response(
            prompt=synthesis_prompt,
            llm_provider="gemini" # or "gpt-oss-20b" depending on your setup
        )
        print("  - English report generated.")

        # Call for Chinese translation
        translation_prompt = f"Please translate the following clinical report into fluent, professional Traditional Chinese and reply with content only:\n\n---\n\n{final_report_en}"
        final_report_zh = llm_response(
            prompt=translation_prompt,
            llm_provider="gemini"
        )
        print("  - Chinese translation complete.")
        
        trace = "Node: Final report synthesis complete."
        return {
            "generated_reports": {"en": final_report_en, "zh": final_report_zh},
            "trace_log": state.get("trace_log", []) + [trace]
        }
    except Exception as e:
        error_message = f"Node (Report Generator) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}