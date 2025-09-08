from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from agents.client.agent_client import create_llm_agent
from agents.sub_agents.act_to_brain.tools.pipeline import pipeline

# -----------------------
# ADK Agent Instructions
# -----------------------

INSTRUCTION = """
You are an fMRI NIfTI processing agent specialized in Alzheimer's disease detection using deep learning and neuroimaging analysis.

Your primary task is to run a complete inference pipeline on fMRI neuroimaging data (.nii.gz files) to:

1. **Model Inspection & Layer Selection**:
   - Analyze the neural network architecture (CapsNet-RNN)
   - Select appropriate layers for activation visualization
   - Validate layer selections using LLM-based filtering

2. **Neural Network Inference**:
   - Load pre-trained model weights
   - Run inference on fMRI data using sliding window approach
   - Extract activations from selected layers
   - Classify the input as AD (Alzheimer's Disease) or CN (Cognitively Normal)

3. **Activation Analysis & Brain Mapping**:
   - Convert neural activations to NIfTI format
   - Resample to AAL3 brain atlas coordinate system
   - Map activations to specific brain regions
   - Generate visualization heatmaps

4. **Results Storage & Clinical Interpretation**:
   - Use save_results function to properly format and store analysis results
   - Analyze activation patterns in context of Alzheimer's pathology
   - Identify key brain regions involved in the classification
   - Provide structured results for clinical report generation

**Expected Output Format**:
Return results as a structured JSON containing:
- Classification result (AD/CN)
- Selected neural network layers with justification
- Brain region activation analysis
- Visualization file paths
- Clinical interpretation summary
- Properly saved results from save_results function

**Key Requirements**:
- Always use the provided pipeline tool to run the complete analysis
- Ensure all processing steps complete successfully including save_results
- Provide clear interpretation of neural activation patterns
- Focus on clinically relevant brain regions (hippocampus, temporal lobe, etc.)
- Generate publication-ready visualizations

The pipeline automatically handles all technical aspects including model loading, data preprocessing, activation extraction, atlas mapping, result saving, and visualization generation.
"""

# -----------------------
# Define ADK Agent
# -----------------------

map_act_brain_agent = create_llm_agent(
    name="MapActBrainAgent",
    # model=LiteLlm(model="ollama_chat/gpt-oss:20b"), 
    # model="gemini-2.5-flash-lite",
    description="Advanced fMRI neuroimaging agent for Alzheimer's disease detection using deep learning, brain activation analysis, and proper results storage.",
    instruction=INSTRUCTION,
    tools=[pipeline],
    output_key="map_act_brain_result",
)


# -----------------------
# Session & Runner Setup
# -----------------------

if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    APP_NAME = "map_act_brain_agent_pipeline"
    USER_ID = "neuroimaging_user"
    SESSION_ID = "nii-session-001"

    async def main():
        """
        Run the NIfTI inference agent with proper ADK session management.
        """
        # Create session service
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=SESSION_ID
        )

        # Create runner
        runner = Runner(
            agent=map_act_brain_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        # Prepare user message
        user_message = types.Content(
            role="user",
            parts=[types.Part(
                text="Run complete fMRI analysis pipeline for Alzheimer's detection. "
                     "Use subject_id='test-patient-001' for this analysis. "
                     "Ensure proper results saving and provide clinical interpretation."
            )],
        )

        print("\n" + "="*80)
        print("STARTING NII INFERENCE AGENT (with save_results integration)")
        print("="*80)
        print(f"App: {APP_NAME}")
        print(f"User: {USER_ID}")
        print(f"Session: {SESSION_ID}")
        print(f"Agent: {map_act_brain_agent.name}")
        print("\n>>> Sending request to map_act_brain_agent...\n")

        # Run the agent
        final_result = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text
                
        print("\n" + "="*80)
        print("ACT TO BRAIN AGENT RESPONSE")
        print("="*80)
        print(final_result)
        print("\n" + "="*80)
        
        return final_result

    # Run the async main function
    result = asyncio.run(main())
