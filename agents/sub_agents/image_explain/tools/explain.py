from pydantic import BaseModel
from agents.llm_client.gemini_client import gemini_image

# --- Input Configuration ---
IMAGE_PATHS = [
    "figures/agent_test/agent_test_capsnet_conv3/activation_map_mosaic.png",
    "figures/group/group_activation_with_dmn.png",
]

INSTRUCTION = """
You are given two fMRI-based brain activation maps:

1. **Individual activation** from a deep learning model analyzing subject sub-14.
2. **Group-level average activation** from patients diagnosed with Alzheimer's disease (AD).

Please analyze the individual activation map in reference to the AD group activation. Return your answer in this structured format:

{
  "text": "<A detailed clinical explanation comparing both maps>",
  "highlighted_regions": ["Region1", "Region2", "..."]
}

Make sure to:
- Identify regions in the individual map that overlap with or diverge from AD group patterns.
- Discuss cognitive functions of those regions.
- Consider implications for Alzheimer's or related disorders.
- Use only medically meaningful and relevant regions.
"""


# --- Structured Response Schema ---
class ActivationExplanation(BaseModel):
    text: str
    highlighted_regions: list[str]


# --- Main Tool Function ---
def explain_activation_map():
    raw_response_str = gemini_image(
        prompt="Analyze the activation maps",
        image_path=IMAGE_PATHS,
        system_instruction=INSTRUCTION,
        mime_type="application/json",
        response_schema=ActivationExplanation,
    )

    # Parse the raw JSON string into an ActivationExplanation object
    response_object = ActivationExplanation.model_validate_json(raw_response_str)

    print("[Gemini Explanation Output]:\n")
    print(raw_response_str)
    print("\n[Highlighted Regions]:", response_object.highlighted_regions)

    return {
        "text": response_object.text,
        "highlighted_regions": response_object.highlighted_regions,
    }


# --- Test run ---
if __name__ == "__main__":
    explain_activation_map()
