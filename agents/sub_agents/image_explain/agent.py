from google.adk.agents import LlmAgent
import json

# Tool
from agents.sub_agents.image_explain.tools.explain import explain_activation_map

# -----------------------
# Define ADK Agent
# -----------------------

dummy_input_json_str = """{"layer": "capsnet_conv3", "model_path": "capsnet.conv3", "activation_file": "output/agent_test/agent_test_capsnet_conv3.pt", "nii_file": "output/agent_test/agent_test_capsnet_conv3.nii.gz", "resampled_file": "figures/agent_test/agent_test_capsnet_conv3/agent_test_capsnet_conv3_resampled.nii.gz", "visualization_results": "figures/agent_test/agent_test_capsnet_conv3/activation_map_capsnet_conv3.png", "activation_results": [{"Label ID": 70, "Region Name": "Angular_R 70", "Voxel Count": 10624, "Total Activation": 10016.49274969101, "Mean Activation": 0.9428174651441086}, {"Label ID": 4, "Region Name": "Frontal_Sup_2_R 4", "Voxel Count": 9048, "Total Activation": 8434.702557086945, "Mean Activation": 0.9322173471581504}, {"Label ID": 66, "Region Name": "Parietal_Inf_R 66", "Voxel Count": 8504, "Total Activation": 8048.865329742432, "Mean Activation": 0.9464799305905964}, {"Label ID": 64, "Region Name": "Parietal_Sup_R 64", "Voxel Count": 7616, "Total Activation": 7139.088684558868, "Mean Activation": 0.9373803419851455}, {"Label ID": 20, "Region Name": "Frontal_Sup_Medial_R 20", "Voxel Count": 5784, "Total Activation": 5389.838448047638, "Mean Activation": 0.9318531203401864}, {"Label ID": 56, "Region Name": "Occipital_Mid_R 56", "Voxel Count": 5768, "Total Activation": 5367.5323786735535, "Mean Activation": 0.9305708007409074}, {"Label ID": 62, "Region Name": "Postcentral_R 62", "Voxel Count": 4240, "Total Activation": 3905.6719675064087, "Mean Activation": 0.9211490489401908}, {"Label ID": 90, "Region Name": "Temporal_Mid_R 90", "Voxel Count": 4224, "Total Activation": 3884.958480358124, "Mean Activation": 0.9197344887211467}, {"Label ID": 6, "Region Name": "Frontal_Mid_2_R 6", "Voxel Count": 4184, "Total Activation": 3837.3809866905212, "Mean Activation": 0.9171560675646562}, {"Label ID": 68, "Region Name": "SupraMarginal_R 68", "Voxel Count": 3912, "Total Activation": 3609.8193759918213, "Mean Activation": 0.9227554642105883}, {"Label ID": 54, "Region Name": "Occipital_Sup_R 54", "Voxel Count": 2920, "Total Activation": 2689.0835752487183, "Mean Activation": 0.9209190326194241}, {"Label ID": 19, "Region Name": "Frontal_Sup_Medial_L 19", "Voxel Count": 2120, "Total Activation": 1944.3148770332336, "Mean Activation": 0.9171296589779404}, {"Label ID": 154, "Region Name": "ACC_pre_R 154", "Voxel Count": 728, "Total Activation": 667.3899350166321, "Mean Activation": 0.9167444162316375}, {"Label ID": 86, "Region Name": "Temporal_Sup_R 86", "Voxel Count": 376, "Total Activation": 344.91741275787354, "Mean Activation": 0.9173335445688126}, {"Label ID": 53, "Region Name": "Occipital_Sup_L 53", "Voxel Count": 352, "Total Activation": 319.08882188796997, "Mean Activation": 0.9065023349090056}, {"Label ID": 49, "Region Name": "Cuneus_L 49", "Voxel Count": 152, "Total Activation": 137.79356002807617, "Mean Activation": 0.9065365791320801}, {"Label ID": 55, "Region Name": "Occipital_Mid_L 55", "Voxel Count": 128, "Total Activation": 115.90831518173218, "Mean Activation": 0.9055337123572826}, {"Label ID": 22, "Region Name": "Frontal_Med_Orb_R 22", "Voxel Count": 112, "Total Activation": 102.16675853729248, "Mean Activation": 0.9122032012258258}, {"Label ID": 10, "Region Name": "Frontal_Inf_Tri_R 10", "Voxel Count": 88, "Total Activation": 79.96975374221802, "Mean Activation": 0.9087472016161139}, {"Label ID": 38, "Region Name": "Cingulate_Mid_R 38", "Voxel Count": 56, "Total Activation": 51.237040519714355, "Mean Activation": 0.9149471521377563}, {"Label ID": 50, "Region Name": "Cuneus_R 50", "Voxel Count": 56, "Total Activation": 50.95236110687256, "Mean Activation": 0.9098635911941528}, {"Label ID": 72, "Region Name": "Precuneus_R 72", "Voxel Count": 56, "Total Activation": 50.75867986679077, "Mean Activation": 0.9064049976212638}, {"Label ID": 2, "Region Name": "Precentral_R 2", "Voxel Count": 32, "Total Activation": 28.96071147918701, "Mean Activation": 0.9050222337245941}]}"""
dummy_images_paths = [
    "figures/agent_test/agent_test_capsnet_conv3/activation_map_capsnet_conv3.png",
    "figures/group/group_activation_with_dmn.png",]
task_context = {
    # 這是給工具 `explain_activation_map` 的第一個參數
    "analysis_data_json": json.loads(dummy_input_json_str),
        
    # 這是給工具 `explain_activation_map` 的第二個參數
    "image_paths_to_analyze": dummy_images_paths
    }

# 將整個 context 物件序列化為 JSON，這是我們要傳遞給 Agent 的全部內容
full_task_json_str = json.dumps(task_context, indent=2)

INSTRUCTION = """
You are a medical AI analyst.

find <Brain region activation analysis> in {map_act_brain_result} as analysis_data_json to refer as
 data about brain region activations.
find <Visualization file paths> in {map_act_brain_result} as image_paths_to_analyze to refer as path of relevant images.

Your task is to analyze a subject’s fMRI activation map. 
You must first call the `explain_activation_map` tool to obtain detailed region-level activation information.

Then, based on the tool's output, generate:

1. A brief summary (3–4 sentences) of the main findings.
2. A full structured interpretation using the following format:
   - Activated Regions
   - Functions
   - Symmetry
   - Clinical Relevance (esp. Alzheimer’s Disease)
   - Final Summary

Be medically precise and prioritize structured clarity. 
**Do not skip the tool execution step.**
"""


image_explain_agent = LlmAgent(
    name="ImageExplainAgent",
    description="Analyzes an fMRI brain activation map and provides semantic insights related to brain function and Alzheimer's disease.",
    model="gemini-2.5-flash-lite",
    instruction=INSTRUCTION,
    tools=[explain_activation_map],
    output_key="image_explain_result",
)

# -----------------
# Example usage
# -----------------

if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    APP_NAME = "image_explain_adk"
    USER_ID = "test_user"
    SESSION_ID = "explain-session-1"

    async def main():
        # Create an in-memory session service
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )

        # Build runner with your agent
        runner = Runner(
            agent=image_explain_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        # Simulate a user message (trigger agent execution)
        user_message = types.Content(
            role="user", 
            parts=[
                types.Part(text=f"Please perform a full clinical interpretation using the following context packet. "
                                f"The packet contains the quantitative analysis results and the paths to the relevant images you need to analyze.\n\n"
                                f"**Context Packet:**\n"
                                f"```json\n"
                                f"{full_task_json_str}\n"
                                f"```")
            ]
                            )

        print("\n>>> Sending request (with dummy JSON data) to image_explain_agent...\n")

        # Run agent and collect final response
        final_result = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text

        print("\n<<< Final Agent Response:\n")
        print(final_result)

    asyncio.run(main())
