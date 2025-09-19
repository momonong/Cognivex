# app/agents/2_dynamic_filtering.py
from app.graph.state import AgentState
# Import tools and constants from our core library
from app.core.fmri_processing.pipeline_steps import dynamic_filtering, OUTPUT_DIR

def filter_layers_dynamically(state: AgentState) -> dict:
    """
    Node 2: Uses an LLM to dynamically filter layers based on activation stats.
    """
    print("\n--- Node: 2. Dynamic Layer Filtering ---")
    subject_id = state['subject_id']
    validated_layers = state.get('validated_layers', [])
    save_name_prefix = f"{subject_id}"

    try:
        keep_entries, _, _ = dynamic_filtering(
            results={},  # This is a temporary dict, not needed from state
            selected_layers=validated_layers,
            activation_dir=OUTPUT_DIR,
            save_name_prefix=save_name_prefix,
            delete_rejected=True,
        )
        
        if not keep_entries:
            print("[Warning] No layers passed filtering. Using first validated layer as fallback.")
            keep_entries = validated_layers[:1]

        trace = f"Node 2: Filtering complete. Kept {len(keep_entries)} layers."

        return {
            "final_layers": keep_entries,
            "trace_log": state.get("trace_log", []) + [trace]
        }

    except Exception as e:
        error_message = f"Node 2 (Filtering) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}