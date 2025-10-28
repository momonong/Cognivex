import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any
# Assuming this is your actual LLM service
from app.services.llm_providers import llm_response 

# ==============================================================================
#  PART 1: The Instruction (System Prompt) - UPDATED
# ==============================================================================

INSTRUCTION = """
## Your Role
You are a meticulous and expert model activation filter assistant. You are analyzing a 2D CNN (ShuffleNet-based).

## Your Core Task
Your primary function is to analyze lists of model layers with their activation statistics and select ONLY the most informative layers suitable for visualization.

## Selection Guidelines
You MUST strictly adhere to the following rules in your analysis:

1.  **Prioritize layers with HIGH activation**:
    - `nonzero_ratio > 0.1` (indicates broad spatial activation).
    - `mean_activation > 0.001` (avoids near-zero useless maps).

2.  **Drop layers with LOW activation**:
    - Any layer with `nonzero_ratio < 0.1` OR `mean_activation < 0.001` MUST be dropped.

3.  **Prioritize Layer Types**:
    - **STRONGLY PREFER** `Conv2d` layers, as they contain the primary feature maps.
    - **DE-PRIORITIZE** `BatchNorm2d` and `AvgPool2d`. Only include them if their activations are *exceptionally* high and the `Conv2d` is not available.
    - **DROP ALL** non-spatial layers like `Linear`, `AdaptiveAvgPool2d`, `Dropout`, or final classifiers.

## Output Format
You MUST provide your response as a single, valid JSON array.
- The array should contain ONLY the layers you have selected.
- Each object in the array must match the `SelectedLayer` schema (keys: `model_path`, `reason`).
- Do not include any other text, explanations, or markdown formatting outside of the final JSON array.
"""

# ==============================================================================
#  PART 2: Pydantic Models for Schema Enforcement
# ==============================================================================

class SelectedLayer(BaseModel):
    """
    Defines the structure of the expected output from the LLM 
    for each selected layer.
    """
    model_path: str = Field(
        ..., description="The exact model path of the layer being selected."
    )
    reason: str = Field(
        ..., description="A brief justification for why this layer was selected."
    )

# The 'ValidationResponse' wrapper class is removed as it's cleaner
# to ask the LLM for a direct list (List[SelectedLayer]).

# ==============================================================================
#  PART 3: The Completed Function
# ==============================================================================

def validate_layers_by_llm(
    layer_stats_list: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Calls llm to decide which layers to keep based on activation stats + semantic metadata.

    Args:
        layer_stats_list (List[Dict[str, Any]]):
            A list of dictionaries, where each dictionary contains a layer's metadata
            (model_path, layer_name, etc.) and its calculated activation statistics.

    Returns:
        List[Dict[str, str]]: A list of dictionaries for the layers that were selected by the LLM.
    """
    if not layer_stats_list:
        return []

    # Build the prompt
    prompt = f"""
        Please analyze the following model layer data based on your established guidelines.

        **Input Data:**
        ```json
        {json.dumps(layer_stats_list, indent=2)}
        ```
        Provide your final selection in the required JSON format (a JSON array).
    """

    try:
        # Call the LLM. 
        # Note: We pass the *type* List[SelectedLayer] as the schema hint.
        response_json_str = llm_response(
            prompt=prompt,
            system_instruction=INSTRUCTION,
            mime_type="application/json",
            response_schema=List[SelectedLayer] # Use the list schema
        )
        
        # Directly parse the JSON string response
        selected_layers = json.loads(response_json_str)
        
        # Ensure it's a list as requested
        if isinstance(selected_layers, list):
            return selected_layers
        else:
            print(f"[Warning] LLM returned an unexpected format (expected list): {type(selected_layers)}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"[Warning] Failed to parse JSON response: {e}")
        print(f"[Debug] Raw response: {response_json_str}")
        return [] # Fail gracefully
        
    except Exception as e:
        print(f"[Warning] LLM validation failed: {e}")
        # Fallback: return all layers that *look* like they have stats
        return [{"model_path": layer.get("model_path", ""), "reason": "LLM validation unavailable"} 
                for layer in layer_stats_list if "mean_activation" in layer]

# ==============================================================================
#  PART 4: Test Block
# ==============================================================================

if __name__ == "__main__":
    
    # --- Mock llm_response function for testing ---
    # This simulates the LLM call so you can run this file directly
    def llm_response(prompt, system_instruction, mime_type, response_schema):
        print("--- [MOCK LLM Call] ---")
        print(f"Strategy: {system_instruction[:50]}...") # Print first 50 chars of strategy
        print("--- [MOCK LLM Response] ---")
        
        # Simulate the LLM's filtered JSON response
        # Note: It correctly filters out '...stage2.0.dwconv' (low stats)
        #       and '...stage4.1.bn3' (BatchNorm)
        mock_json_response = [
            {
                "model_path": "backbone.stage4.1.gconv2",
                "reason": "Selected: High activation Conv2d layer."
            }
        ]
        return json.dumps(mock_json_response, indent=2)
    # --- End Mock Function ---
    
    # This is a MOCK list, simulating the output of the *next* step
    # (after activation stats have been calculated)
    mock_layer_stats_list = [
        {
            "model_path": "backbone.stage2.0.dwconv",
            "layer_type": "Conv2d",
            "mean_activation": 0.0001,  # -> Should be FILTERED OUT (too low)
            "nonzero_ratio": 0.05
        },
        {
            "model_path": "backbone.stage4.1.bn3",
            "layer_type": "BatchNorm2d",
            "mean_activation": 0.5,
            "nonzero_ratio": 0.9       # -> Should be FILTERED OUT (BatchNorm)
        },
        {
            "model_path": "backbone.stage4.1.gconv2",
            "layer_type": "Conv2d",
            "mean_activation": 0.2,    # -> Should be KEPT
            "nonzero_ratio": 0.8
        },
        {
            "model_path": "fc_classify",
            "layer_type": "Linear",
            "mean_activation": 0.6,    # -> Should be FILTERED OUT (Linear)
            "nonzero_ratio": 1.0
        }
    ]
    
    print(f"Input layers for filtering: {len(mock_layer_stats_list)}")
    
    # Call the function we are testing
    filtered_layers = validate_layers_by_llm(mock_layer_stats_list)
    
    print("\n--- [Filter Result] ---")
    print(json.dumps(filtered_layers, indent=2))
    
    if len(filtered_layers) == 1 and filtered_layers[0]["model_path"] == "backbone.stage4.1.gconv2":
        print("\n[SUCCESS] Filter correctly selected the 'gconv2' layer.")
    else:
        print("\n[FAILURE] Filter did not return the expected layer.")