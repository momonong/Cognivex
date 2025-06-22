from google.adk.agents.llm_agent import Agent
from agents.sub_agents.nii_inference.tools.full_pipeline import run_full_inference_pipeline

INSTRUCTION = """
You are an fMRI NIfTI processing agent. You must run the full inference pipeline on a given .nii.gz brain image.

Your tasks include:
1. Model inspection and layer selection
2. Attaching hook and running inference
3. Gemini-based activation filtering
4. Converting activations to NIfTI
5. Resampling to AAL3 atlas
6. Analyzing activation brain regions
7. Producing visualizations

Always return the result in this format:
{
  "classification": "<AD or CN>",
  "final_layers": ["conv3", "fc1"],
  "activation_results": [
    {
      "layer": "conv3",
      "summary": "This activation involves regions such as the hippocampus and posterior cingulate...",
      "visualization_path": "figures/..."
    },
    ...
  ]
}
"""

nii_inference_agent = Agent(
    name="nii_inference_agent",
    model="gemini-2.5-flash",
    description="Agent for full NIfTI fMRI inference and semantic analysis.",
    instruction=INSTRUCTION,
    tools=[run_full_inference_pipeline],
    output_key="activation_results",  # 或改成 return 完整物件
)
