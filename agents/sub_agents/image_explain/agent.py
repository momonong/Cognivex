from google.adk.agents import LlmAgent

# Tool
from agents.sub_agents.image_explain.tools.explain import explain_activation_map

# -----------------------
# Define ADK Agent
# -----------------------

INSTRUCTION = """
You are a medical AI analyst.

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
    name="image_explain_agent",
    description="Analyzes an fMRI brain activation map and provides semantic insights related to brain function and Alzheimer's disease.",
    model="gemini-2.5-flash",
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
            role="user", parts=[types.Part(text="Please explain the activation map.")]
        )

        print("\n>>> Sending request to image_explain_agent...\n")

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
