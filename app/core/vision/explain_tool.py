# app/core/vision/explain_tool.py

from pydantic import BaseModel
from typing import List, Dict, Any
# 確保 import 路徑正確
from app.services.llm_provider import gemini_image 


INSTRUCTION = """
You are an expert neuroradiologist AI assistant. Your task is to provide a detailed, clinically relevant explanation of an fMRI activation map by synthesizing structured data and visual information.

You will be given two pieces of information:
1.  **ANALYSIS_DATA (JSON)**: A structured JSON report from a previous analysis pipeline. This contains the ground truth about activated brain regions (e.g., 'Angular_R').
2.  **IMAGES**: A list of image files for visual context.

**CRITICAL DIRECTIVE: YOUR PRIMARY SOURCE OF TRUTH FOR ALL FACTUAL CLAIMS (especially Left/Right hemisphere) IS THE `ANALYSIS_DATA` JSON. YOU MUST TRUST THE JSON DATA OVER YOUR OWN VISUAL INTERPRETATION.**

Your workflow is:
1.  **Fact Extraction**: Carefully read the `ANALYSIS_DATA` JSON to understand the key activated regions and their laterality (`_R` for Right, `_L` for Left).
2.  **Visual Correlation**: Look at the provided IMAGES only to describe the visual pattern and intensity of the activation clusters identified in Step 1.
3.  **Synthesize Report**: Combine the facts and visual context to write a comprehensive explanation, returning it in the required JSON format.
"""

# --- Structured Response Schema ---
class ActivationExplanation(BaseModel):
    text: str
    highlighted_regions: list[str]


# --- Main Tool Function ---
def explain_activation_map(
    image_paths: List[str],
    analysis_data_json: str,
    classification_result: str,
    subject_id: str
) -> Dict[str, Any]:    
    """
    Analyzes fMRI activation maps by synthesizing structured data and visual information.
    """
    print("  - Calling Gemini Vision to explain activation map...")
    prompt = f"""
    Please analyze the provided fMRI data and images according to my instructions.

    **SUBJECT ID:**
    {subject_id}

    **OVERALL CLASSIFICATION (from primary model):**
    {classification_result}

    **ANALYSIS_DATA (Source of Truth):**
    ```json
    {analysis_data_json}
    ```
    IMAGES:
    Image 1 is the individual activation map.
    Image 2 is the group-level average activation map.

    Now, using the OVERALL CLASSIFICATION as crucial context, generate the detailed clinical explanation in the required JSON format.
    """
    
    raw_response_str = gemini_image(
        prompt=prompt,
        image_path=image_paths,
        system_instruction=INSTRUCTION,
        mime_type="application/json",
        response_schema=ActivationExplanation,
    )

    # Parse the raw JSON string into an ActivationExplanation object
    response_object = ActivationExplanation.model_validate_json(raw_response_str)
    print("  - Gemini Vision analysis complete.")
    
    return {
        "text": response_object.text,
        "highlighted_regions": response_object.highlighted_regions,
    }