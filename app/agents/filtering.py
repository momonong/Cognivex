# app/agents/2_dynamic_filtering.py
from app.graph.state import AgentState

def filter_layers_dynamically(state: AgentState) -> dict:
    """
    Node 2: Simplified layer filtering for ShuffleNet integration.
    Since the new generic pipeline handles layer selection automatically,
    this node now just passes through the validated layers.
    """
    print("\n--- Node: 2. Layer Filtering (Simplified for ShuffleNet) ---")
    
    validated_layers = state.get('validated_layers', [])
    
    if not validated_layers:
        print("[Warning] No validated layers found. This may indicate an issue with the inference step.")
        return {"error_log": state.get("error_log", []) + ["No validated layers found for filtering"]}
    
    # For ShuffleNet, we typically want to keep all validated layers
    # since the layer selection is already optimized in the generic pipeline
    final_layers = validated_layers
    
    print(f"  - Keeping {len(final_layers)} validated layers for further processing")
    for layer in final_layers:
        layer_name = layer.get('model_path', 'Unknown')
        print(f"    * {layer_name}")
    
    trace = f"Node 2: Layer filtering complete. Kept {len(final_layers)} layers."

    return {
        "final_layers": final_layers,
        "trace_log": state.get("trace_log", []) + [trace]
    }