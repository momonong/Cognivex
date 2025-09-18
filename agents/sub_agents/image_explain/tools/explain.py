from pydantic import BaseModel
from agents.client.llm_client import gemini_image
from typing import List, Dict, Any

# --- Input Configuration ---
IMAGE_PATHS = [
    "figures/agent_test/agent_test_capsnet_conv3/activation_map_capsnet_conv3.png",
    "figures/group/group_activation_with_dmn.png",
]

DUMMY_ANALYSIS_DATA_JSON = """
{'layer': 'capsnet_conv3', 'model_path': 'capsnet.conv3', 'activation_file': 'output/agent_test/agent_test_capsnet_conv3.pt', 'nii_file': 'output/agent_test/agent_test_capsnet_conv3.nii.gz', 'resampled_file': 'figures/agent_test/agent_test_capsnet_conv3/agent_test_capsnet_conv3_resampled.nii.gz', 'visualization_results': 'figures/agent_test/agent_test_capsnet_conv3/activation_map_capsnet_conv3.png', 'activation_results': [{'Label ID': 70, 'Region Name': 'Angular_R 70', 'Voxel Count': 10624, 'Total Activation': 10016.49274969101, 'Mean Activation': 0.9428174651441086}, {'Label ID': 4, 'Region Name': 'Frontal_Sup_2_R 4', 'Voxel Count': 9048, 'Total Activation': 8434.702557086945, 'Mean Activation': 0.9322173471581504}, {'Label ID': 66, 'Region Name': 'Parietal_Inf_R 66', 'Voxel Count': 8504, 'Total Activation': 8048.865329742432, 'Mean Activation': 0.9464799305905964}, {'Label ID': 64, 'Region Name': 'Parietal_Sup_R 64', 'Voxel Count': 7616, 'Total Activation': 7139.088684558868, 'Mean Activation': 0.9373803419851455}, {'Label ID': 20, 'Region Name': 'Frontal_Sup_Medial_R 20', 'Voxel Count': 5784, 'Total Activation': 5389.838448047638, 'Mean Activation': 0.9318531203401864}, {'Label ID': 56, 'Region Name': 'Occipital_Mid_R 56', 'Voxel Count': 5768, 'Total Activation': 5367.5323786735535, 'Mean Activation': 0.9305708007409074}, {'Label ID': 62, 'Region Name': 'Postcentral_R 62', 'Voxel Count': 4240, 'Total Activation': 3905.6719675064087, 'Mean Activation': 0.9211490489401908}, {'Label ID': 90, 'Region Name': 'Temporal_Mid_R 90', 'Voxel Count': 4224, 'Total Activation': 3884.958480358124, 'Mean Activation': 0.9197344887211467}, {'Label ID': 6, 'Region Name': 'Frontal_Mid_2_R 6', 'Voxel Count': 4184, 'Total Activation': 3837.3809866905212, 'Mean Activation': 0.9171560675646562}, {'Label ID': 68, 'Region Name': 'SupraMarginal_R 68', 'Voxel Count': 3912, 'Total Activation': 3609.8193759918213, 'Mean Activation': 0.9227554642105883}, {'Label ID': 54, 'Region Name': 'Occipital_Sup_R 54', 'Voxel Count': 2920, 'Total Activation': 2689.0835752487183, 'Mean Activation': 0.9209190326194241}, {'Label ID': 19, 'Region Name': 'Frontal_Sup_Medial_L 19', 'Voxel Count': 2120, 'Total Activation': 1944.3148770332336, 'Mean Activation': 0.9171296589779404}, {'Label ID': 154, 'Region Name': 'ACC_pre_R 154', 'Voxel Count': 728, 'Total Activation': 667.3899350166321, 'Mean Activation': 0.9167444162316375}, {'Label ID': 86, 'Region Name': 'Temporal_Sup_R 86', 'Voxel Count': 376, 'Total Activation': 344.91741275787354, 'Mean Activation': 0.9173335445688126}, {'Label ID': 53, 'Region Name': 'Occipital_Sup_L 53', 'Voxel Count': 352, 'Total Activation': 319.08882188796997, 'Mean Activation': 0.9065023349090056}, {'Label ID': 49, 'Region Name': 'Cuneus_L 49', 'Voxel Count': 152, 'Total Activation': 137.79356002807617, 'Mean Activation': 0.9065365791320801}, {'Label ID': 55, 'Region Name': 'Occipital_Mid_L 55', 'Voxel Count': 128, 'Total Activation': 115.90831518173218, 'Mean Activation': 0.9055337123572826}, {'Label ID': 22, 'Region Name': 'Frontal_Med_Orb_R 22', 'Voxel Count': 112, 'Total Activation': 102.16675853729248, 'Mean Activation': 0.9122032012258258}, {'Label ID': 10, 'Region Name': 'Frontal_Inf_Tri_R 10', 'Voxel Count': 88, 'Total Activation': 79.96975374221802, 'Mean Activation': 0.9087472016161139}, {'Label ID': 38, 'Region Name': 'Cingulate_Mid_R 38', 'Voxel Count': 56, 'Total Activation': 51.237040519714355, 'Mean Activation': 0.9149471521377563}, {'Label ID': 50, 'Region Name': 'Cuneus_R 50', 'Voxel Count': 56, 'Total Activation': 50.95236110687256, 'Mean Activation': 0.9098635911941528}, {'Label ID': 72, 'Region Name': 'Precuneus_R 72', 'Voxel Count': 56, 'Total Activation': 50.75867986679077, 'Mean Activation': 0.9064049976212638}, {'Label ID': 2, 'Region Name': 'Precentral_R 2', 'Voxel Count': 32, 'Total Activation': 28.96071147918701, 'Mean Activation': 0.9050222337245941}]}
"""

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
) -> Dict[str, Any]:    
    """
    Analyzes fMRI activation maps by synthesizing structured data and visual information.

    This tool takes quantitative data from a previous analysis pipeline as the "source of truth"
    and correlates it with the provided images to generate a detailed clinical explanation.

    Args:
        image_paths (List[str]): A list of paths to the image files to be analyzed.
                                 Typically [individual_map_path, group_map_path].
        analysis_data_json (str): The output from the nii_inference pipeline,
                                    containing factual data about activated regions.

    Returns:
        Dict[str, Any]: A dictionary containing the detailed text explanation and a list of highlighted regions.
    """
    print("IMAGE EXPLAIN START")
    prompt = f"""
    Please analyze the provided fMRI data and images according to my instructions.

    **ANALYSIS_DATA (Source of Truth):**
    ```json
    {analysis_data_json}
    ```
    IMAGES:

    Image 1 is the individual activation map.

    Image 2 is the group-level average activation map.

    Now, generate the detailed clinical explanation in the required JSON format.
    """
    # # print("IMAGES TO ANALYZE:", image_paths)
    # # print("ANALYSIS_DATA JSON:", analysis_data_json)

    raw_response_str = gemini_image(
        prompt=prompt,
        image_path=image_paths,
        system_instruction=INSTRUCTION,
        mime_type="application/json",
        response_schema=ActivationExplanation,
    )

    # Parse the raw JSON string into an ActivationExplanation object
    response_object = ActivationExplanation.model_validate_json(raw_response_str)

    print("[Gemini Explanation Output]:\n")
    print(raw_response_str)
    print("\n[Highlighted Regions]:", response_object.highlighted_regions)
    print("IMAGE EXPLAIN END")
    return {
        "text": response_object.text,
        "highlighted_regions": response_object.highlighted_regions,
    }


# --- Test run ---
if __name__ == "__main__":
    IMAGE_PATH = "figures/agent_test/agent_test_capsnet_conv3/activation_map_capsnet_conv3.png"
    ANALYSIS_DATA_JSON = "The fMRI analysis for subject sub-14 indicates Alzheimer's Disease (AD). The inference was performed using the CapsNet-RNN model, with activations extracted from the `capsnet.conv3` layer.\n\n**Activation Analysis:**\n\nThe analysis revealed significant activation patterns in several brain regions:\n\n*   **Right Hemisphere Dominance:** Higher activations were observed in the right hemisphere, particularly in the **Angular gyrus (Angular_R)**, **Superior Frontal gyrus (Frontal_Sup_2_R)**, **Inferior Parietal gyrus (Parietal_Inf_R)**, and **Superior Parietal gyrus (Parietal_Sup_R)**.\n*   **Parietal and Temporal Lobe Involvement:** The **Parietal_Inf_R**, **Parietal_Sup_R**, and **Temporal_Mid_R** regions showed notable activation, which is consistent with known patterns of neurodegeneration in AD affecting these areas.\n*   **Frontal Lobe Engagement:** Activations were also detected in the **Frontal_Sup_2_R**, **Frontal_Sup_Medial_R**, and **Frontal_Mid_2_R**, suggesting involvement of frontal cognitive networks.\n*   **Other Regions:** Moderate activations were observed in the **Postcentral_R**, **Occipital_Mid_R**, **SupraMarginal_R**, **Occipital_Sup_R**, **Frontal_Sup_Medial_L**, **ACC_pre_R**, and **Temporal_Sup_R**.\n\n**Clinical Interpretation:**\n\nThe observed activation patterns, particularly in the parietal and temporal lobes, align with the typical progression of Alzheimer's disease, which often involves atrophy and functional changes in these regions crucial for memory, language, and spatial processing. The involvement of frontal regions further supports the widespread network dysfunction seen in AD.\n\n**Visualization:**\n\nThe activation map for the `capsnet.conv3` layer is available at: `figures/agent_test/agent_test_capsnet_conv3/activation_map_capsnet_conv3.png`.\n\nThe detailed results, including brain region activations and file paths, have been saved and are available in the JSON output."
    explain_activation_map(image_paths=[IMAGE_PATH], analysis_data_json=ANALYSIS_DATA_JSON)
