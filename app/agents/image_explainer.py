# app/agents/image_explainer.py
import json
from app.graph.state import AgentState
# 從我們的 core 工具庫中 import 工具
from app.core.vision.explain_tool import explain_activation_map

def explain_image(state: AgentState) -> dict:
    """
    Node: Calls the core explain_activation_map tool with rich context
    to get a primary explanation of the fMRI visualization.
    """
    print("\n--- Node: Image Explainer ---")
    
    # 從 state 中獲取所有需要的上下文資訊
    image_paths = state.get("visualization_paths")
    analysis_data = state.get("activated_regions", [])
    classification = state.get("classification_result", "N/A")
    subject_id = state.get("subject_id", "N/A")
    
    analysis_data_json = json.dumps(analysis_data, indent=2)

    if not image_paths or not analysis_data:
        print("  - Missing image paths or analysis data. Skipping.")
        return {}

    print(f"  - Calling explain_activation_map tool for subject: {subject_id}")
    try:
        # 將豐富的上下文傳遞給工具
        explanation_result = explain_activation_map(
            image_paths=image_paths,
            analysis_data_json=analysis_data_json,
            classification_result=classification, # <-- 傳遞新資訊
            subject_id=subject_id,              # <-- 傳遞新資訊
        )
        
        trace = "Node: Image explanation tool call successful."
        
        return {
            "image_explanation": explanation_result,
            "trace_log": state.get("trace_log", []) + [trace]
        }
    except Exception as e:
        error_message = f"Node (Image Explainer) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}