# app/graph/workflow.py

from re import S
from langgraph.graph import StateGraph, START, END
from .state import AgentState

# Import our NEW, REAL agent nodes
from app.agents.inference import run_inference_and_classification
from app.agents.filtering import filter_layers_dynamically
from app.agents.postprocessing import run_post_processing
from app.agents.entity_linking import link_entities
from app.agents.knowledge_reasoning import enrich_with_knowledge_graph
from app.agents.image_explainer import explain_image
from app.agents.report_generator import generate_final_report

# Import other nodes for the full pipeline later
# from app.agents.node_4_knowledge_reasoning import run_knowledge_reasoning
# from app.agents.node_5_report_generation import generate_report

# Create a new StateGraph with our AgentState
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("inference", run_inference_and_classification)
workflow.add_node("filtering", filter_layers_dynamically)
workflow.add_node("post_processing", run_post_processing)
workflow.add_node("entity_linker", link_entities) # <-- 2. Add new node
workflow.add_node("knowledge_reasoner", enrich_with_knowledge_graph)
workflow.add_node("image_explainer", explain_image)
workflow.add_node("report_generator", generate_final_report)

# Define the edges for the workflow
workflow.add_edge(START, "inference")
workflow.add_edge("inference", "filtering")
workflow.add_edge("filtering", "post_processing")
workflow.add_edge("post_processing", "entity_linker")
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