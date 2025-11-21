from pydantic import BaseModel
# This should be your actual LLM service. 
# If it's not at this path, update the import.
from app.services.llm_providers import llm_response 
import json # Import json for validation

# Strategy-based layer selection instructions
STRATEGY_INSTRUCTIONS = {
    
    # --- NEW V3 STRATEGY ---
    "shufflenet_focused_v3": """
You are an expert model analysis assistant for a modified ShuffleNetV1 model.

Selection Criteria (Strict V3):
1.  Your goal is to find the **single best layer** for Grad-CAM visualization.
2.  This layer MUST be from the **`backbone`** module.
3.  It MUST be the **final *Conv2d* layer** from the **final stage** of the `backbone`.
4.  Algorithm:
    a. Look at the `model_path` list. Find all layers in `backbone.stage4`.
    b. Within `backbone.stage4`, find the block with the **highest index** (e.g., `backbone.stage4.1`).
    c. From that final block (e.g., `stage4.1`), select the **final `Conv2d` layer** (e.g., `backbone.stage4.1.gconv2`).
5.  **DO NOT** select `BatchNorm2d` or `AvgPool2d` layers.
6.  **DO NOT** select layers from the *start* of a stage (e.g., `...stage3.0.gconv1`).
7.  Return **only one** layer.

Output Format:
Return a **valid JSON array** with *exactly one* layer:
[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "Conv2d", 
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification>"
  }
]
""",
    
    # --- Old strategies (kept for reference) ---
    "shufflenet_focused_v2": """
You are an expert model analysis assistant for a modified ShuffleNetV1 model.
... (content from your script) ...
""",
    "shufflenet_focused": """
You are a model analysis assistant for a modified ShuffleNetV1 model (PaperModel).
... (content from your script) ...
""",
    "capsule_focused": """
You are a model analysis assistant for Capsule Networks. Your task is to select the most informative layers for visualizing capsule-based activations.
... (content from your script) ...
""",
    "conv_focused": """
You are a model analysis assistant for Convolutional Networks. Your task is to select the most informative **spatial feature extraction layers** for visualizing activations.
... (content from your script) ...
""",
    "default": """
You are a model analysis assistant. Your task is to select the most informative **spatial feature extraction layer(s)** for visualizing activations (e.g., with GradCAM or 3D attention maps).
... (content from your script) ...
""",
    "improved_capsule": """
You are a model analysis assistant for Capsule Networks with IMPROVED selection criteria based on visualization analysis.
... (content from your script) ...
""",
    "improved_conv": """
You. are a model analysis assistant for Convolutional Networks with IMPROVED selection criteria.
... (content from your script) ...
"""
}


class LayerSelection(BaseModel):
    layer_name: str  
    layer_type: str  
    model_path: str  
    reason: str  


def select_visualization_layers(layers: list[dict], strategy: str = "default") -> str:
    """
    Select visualization layers using specified strategy.
    
    Args:
        layers: List of layer information from model inspection
        strategy: Selection strategy ('shufflenet_focused_v3', 'capsule_focused', etc.)
        
    Returns:
        JSON string of selected layers
    """
    # Get strategy-specific instruction
    instruction = STRATEGY_INSTRUCTIONS.get(strategy, STRATEGY_INSTRUCTIONS["default"])
    
    prompt = (
        f"The model layers are:\n{layers}\n\nPlease select layers for visualization."
    )

    # This calls your actual LLM service
    return llm_response(
        prompt=prompt,
        system_instruction=instruction,
        mime_type="application/json",
        response_schema=list[LayerSelection],
    )


if __name__ == "__main__":
    
    # Import from the new centralized loader and inspector
    # (Update paths if necessary for your execution context)
    from app.core.mri_processing.mri_model_loader import get_model_and_input_shape
    from app.core.mri_processing.pipelines.inspector import inspect_torch_model
    import json

    # 1. Load the correct model
    MODEL, input_shape = get_model_and_input_shape()
    
    if MODEL:
        # 2. Inspect the correct model
        print(f"Inspecting {MODEL.__class__.__name__}...")
        layers = inspect_torch_model(MODEL)
        
        # 3. Call the layer selector with the NEW V3 strategy
        print(f"Selecting layers with 'shufflenet_focused_v3' strategy...")
        response_str = select_visualization_layers(layers, strategy="shufflenet_focused_v3")
        
        # 4. Print the result
        print("\n--- [LLM Response] ---")
        print(response_str)
        
        # 5. Validate JSON
        try:
            json.loads(response_str)
            print("\nValidation: Response is valid JSON.")
        except json.JSONDecodeError:
            print("\nValidation Error: Response is NOT valid JSON.")
            
    else:
        print("Could not load model, skipping selection.")