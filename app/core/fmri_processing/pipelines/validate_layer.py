import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.services.llm_provider import llm_response

# ==============================================================================
#  PART 1: The Instruction (System Prompt) - AI 的工作手冊
# ==============================================================================

INSTRUCTION = """
## Your Role
You are a meticulous and expert model activation filter assistant.

## Your Core Task
Your primary function is to analyze lists of model layers with their activation statistics and select the most informative and meaningful layers suitable for visualization.

## Selection Guidelines
You MUST strictly adhere to the following rules in your analysis:

1.  **Favor layers with**:
    - `nonzero_ratio > 0.1` (indicates broad spatial activation).
    - `mean_activation > 0.001` (avoids near-zero useless maps).

2.  **Drop layers with**:
    - Extremely sparse or weak activations.
    - Non-spatial layers like `Linear`, `GlobalPool`, or final classifiers.

3.  **Special Rule**: `Capsule` layers are allowed **only** if they appear to retain spatial structure.

## Output Format
You MUST provide your response as a single, valid JSON array.
- The array should contain ONLY the layers you have selected.
- Each object in the array must have two keys: `model_path` (string) and `reason` (string, a brief justification for your selection).
- Do not include any other text, explanations, or markdown formatting outside of the final JSON array.
"""

# ==============================================================================
#  PART 2: Pydantic Models for Schema Enforcement
# ==============================================================================


class SelectedLayer(BaseModel):
    """
    Defines the structure of the expected output from the LLM for each selected layer.
    """

    model_path: str = Field(
        ..., description="The exact model path of the layer being selected."
    )
    reason: str = Field(
        ..., description="A brief justification for why this layer was selected."
    )


class ValidationResponse(BaseModel):
    """
    Response wrapper for the list of selected layers.
    This helps with Gemini API schema compatibility.
    """
    selected_layers: List[SelectedLayer] = Field(
        ..., description="List of layers selected for visualization"
    )


# ==============================================================================
#  PART 3: The Completed Function
# ==============================================================================


def validate_layers_by_llm(
    layer_stats_list: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Calls llm to decide which layers to keep based on activation stats + semantic metadata.

    This function separates the static instructions (who the agent is, its rules) from the
    dynamic prompt (the specific data to process).

    Args:
        layer_stats_list (List[Dict[str, Any]]):
            A list of dictionaries, where each dictionary contains a layer's metadata
            (model_path, layer_name, etc.) and its calculated activation statistics.

    Returns:
        List[Dict[str, str]]: A list of dictionaries for the layers that were selected by the LLM,
                              each containing 'model_path' and 'reason'.
    """
    if not layer_stats_list:
        # print("[Warning] No layer stats provided to filter. Returning empty list.")
        return []

    # Build Gemini prompt - 簡潔、直接，只包含數據和行動指令
    prompt = f"""
        Please analyze the following model layer data based on your established guidelines.

        **Input Data:**
        ```json
        {json.dumps(layer_stats_list, indent=2)}
        Provide your final selection in the required JSON format.
    """

    # Gemini response - 不使用 response_schema 來避免相容性問題
    try:
        response_json_str = llm_response(
            prompt=prompt,
            system_instruction=INSTRUCTION,
            mime_type="application/json",
            # 不使用 response_schema 避免相容性問題
        )
        
        # 直接解析 JSON 響應
        validation_response = json.loads(response_json_str)
        
        # 確保返回格式正確
        if isinstance(validation_response, list):
            return validation_response
        elif isinstance(validation_response, dict) and "selected_layers" in validation_response:
            return validation_response["selected_layers"]
        else:
            # print(f"[Warning] Unexpected response format: {type(validation_response)}")
            return validation_response if isinstance(validation_response, list) else []
            
    except json.JSONDecodeError as e:
        # print(f"[Warning] Failed to parse JSON response: {e}")
        # print(f"[Debug] Raw response: {response_json_str}")
        return [{"model_path": layer.get("model_path", ""), "reason": "JSON parsing failed"} for layer in layer_stats_list]
        
    except Exception as e:
        # print(f"[Warning] LLM validation failed: {e}")
        # print("[Info] Falling back to basic validation (returning all input layers)")
        # 如果 LLM 驗證失敗，返回原始輸入作為後備
        return [{"model_path": layer.get("model_path", ""), "reason": "LLM validation unavailable"} for layer in layer_stats_list]

