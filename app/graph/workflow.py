# app/graph/workflow.py

from re import S
from langgraph.graph import StateGraph, START, END
from .state import AgentState

# Import functional MRI agent nodes
from app.agents.inference import run_inference_and_classification
from app.agents.filtering import filter_layers_dynamically
from app.agents.postprocessing import run_post_processing
from app.agents.entity_linking import link_entities
from app.agents.knowledge_reasoning import enrich_with_knowledge_graph
from app.agents.image_explainer import explain_image
from app.agents.report_generator import generate_final_report

# Import structural MRI agent nodes
from app.agents.structural_mri_inference import run_structural_mri_inference
from app.agents.structural_feature_analyzer import analyze_feature_importance
from app.agents.structural_visualizer import generate_structural_visualizations

# Router function for analysis mode
def route_by_analysis_mode(state: AgentState) -> str:
    """
    Route to appropriate inference node based on analysis mode
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node to execute
    """
    mode = state.get("analysis_mode", "functional")
    
    if mode == "structural":
        print(f"\n🔀 Router: Directing to STRUCTURAL MRI branch")
        return "structural_mri_inference"
    else:
        print(f"\n🔀 Router: Directing to FUNCTIONAL MRI branch")
        return "inference"

# Create a new StateGraph with our AgentState
workflow = StateGraph(AgentState)

# Add functional MRI nodes
workflow.add_node("inference", run_inference_and_classification)
workflow.add_node("filtering", filter_layers_dynamically)
workflow.add_node("post_processing", run_post_processing)

# Add structural MRI nodes
workflow.add_node("structural_mri_inference", run_structural_mri_inference)
workflow.add_node("structural_feature_analyzer", analyze_feature_importance)
workflow.add_node("structural_visualizer", generate_structural_visualizations)

# Add shared nodes (used by both branches)
workflow.add_node("entity_linker", link_entities)
workflow.add_node("knowledge_reasoner", enrich_with_knowledge_graph)
workflow.add_node("image_explainer", explain_image)
workflow.add_node("report_generator", generate_final_report)

# Define the edges for the workflow
# Start with conditional routing
workflow.add_conditional_edges(
    START,
    route_by_analysis_mode,
    {
        "structural_mri_inference": "structural_mri_inference",
        "inference": "inference"
    }
)

# === Functional MRI Branch (existing) ===
workflow.add_edge("inference", "filtering")
workflow.add_edge("filtering", "post_processing")
workflow.add_edge("post_processing", "entity_linker")

# === Structural MRI Branch (new) ===
workflow.add_edge("structural_mri_inference", "structural_feature_analyzer")
workflow.add_edge("structural_feature_analyzer", "structural_visualizer")
workflow.add_edge("structural_visualizer", "entity_linker")

# === Shared path (both branches converge) ===
workflow.add_edge("entity_linker", "knowledge_reasoner")
workflow.add_edge("knowledge_reasoner", "image_explainer")
workflow.add_edge("image_explainer", "report_generator")
workflow.add_edge("report_generator", END)

# Compile the graph into a runnable LangChain object
app = workflow.compile()

if __name__ == "__main__":
    import json

    # 1. Define the initial input for the graph
    #    This dictionary must have the keys required by the first node.
    subject_id = "sub_01"
    
    initial_state = {
        "subject_id": subject_id,
        "fmri_scan_path": "data/raw/CN/sub-01/dswausub-009_S_0751_task-rest_bold.nii.gz", 
        "model_path": "model/capsnet/best_capsnet_rnn.pth", 
        "trace_log": [],
        "error_log": [],
    }

    print("="*30)
    print(f"🚀 Starting pipeline run for subject: {subject_id}")
    print("="*30)

    # 2. Execute the graph using the .stream() method
    #    .stream() allows us to see the output of each node as it runs.
    final_state = app.invoke(initial_state)
    print("\n" + "="*30)
    print("✅ Pipeline run finished! Inspecting final state...")
    print("="*30)

    # 3. 使用 json.dumps 美化輸出，讓我們能清楚地看到所有欄位
    if final_state:
        # ensure_ascii=False 確保中文字符能正確顯示
        # indent=2 讓 JSON 格式更易讀
        print(json.dumps(final_state, indent=2, ensure_ascii=False))
    else:
        print("Pipeline did not produce a final state.")