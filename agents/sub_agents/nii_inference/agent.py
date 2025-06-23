from google.adk.agents import LlmAgent

# Tool
from agents.sub_agents.nii_inference.tools.full_pipeline import pipeline

# -----------------------
# Define ADK Agent
# -----------------------

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
  "classification": "AD or CN",
  "final_layers": [
    {
      "model_path": "capsnet.conv3",
      "reason": "Selected because it has high nonzero activation and provides meaningful high-level spatial features."
    },
    ...
  ],
  "activation_results": [
    {
      "layer": "conv3",
      "summary": "This activation map shows significant involvement in the right angular gyrus, superior frontal, and precuneus regions, often linked to early Alzheimer's pathology.",
      "visualization_path": "figures/agent_test/activation_map_mosaic.png"
    },
    ...
  ]
}
"""

nii_inference_agent = LlmAgent(
    name="nii_inference_agent",
    model="gemini-2.5-flash",
    description="Agent for full NIfTI fMRI inference and semantic analysis.",
    instruction=INSTRUCTION,
    tools=[pipeline],
    output_key="nii_inference_result",
)

# -----------------------
# Set up session & runner
# -----------------------


if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    APP_NAME = "nii_inference_adk"
    USER_ID = "test_user"
    SESSION_ID = "nii-session-1"

    async def main():
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )

        runner = Runner(
            agent=nii_inference_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        user_message = types.Content(
            role="user",
            parts=[types.Part(text="Run full NIfTI inference on the provided data.")],
        )

        print("\n>>> Sending request to nii_inference_agent...\n")

        final_result = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text

        print("\n<<< Agent’s final response:\n", final_result)

    asyncio.run(main())
