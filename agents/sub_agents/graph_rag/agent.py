from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from agents.client.agent_client import create_llm_agent
from agents.sub_agents.graph_rag.tools.graphrag import graphrag
from agents.sub_agents.graph_rag.tools.entity_linker import entity_linker_tool

INSTRUCTIONS = """
You are a Research Analyst Agent. Your mission is to use a two-step toolchain to enrich a report.

**Your Context:**
You will receive a `highlighted_regions` list with potentially imprecise names.

**Your Mandate (Follow this order STRICTLY):**

**Step 1: Entity Linking.**
- You MUST first call the `entity_linker_tool`.
- The `dirty_region_list` parameter for this tool MUST be the `highlighted_regions` from your input.

**Step 2: Knowledge Query.**
- After you receive the 'clean list' from the `entity_linker_tool`, you MUST call the `query_regions_by_name` tool.
- The `region_names` parameter for this tool MUST be the 'clean list' you just received.

**Step 3: Final Report.**
- After `query_regions_by_name` returns the final data, you MUST synthesize all information into the final Markdown report as previously instructed.
"""

graph_rag_agent = LlmAgent(
    name="GraphRAGAgent",
    # model=LiteLlm(model="ollama_chat/gpt-oss:20b"),  
    model="gemini-2.5-flash-lite",
    description="Agent for querying and reasoning over knowledge graphs using RAG.",
    instruction=INSTRUCTIONS,
    tools=[entity_linker_tool, graphrag],
    output_key="graph_rag_result",
)


# -----------------------
# Set up session & runner
# -----------------------


if __name__ == "__main__":
    import json
    import asyncio
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types

    APP_NAME = "graph_rag_adk"
    USER_ID = "test_user"
    SESSION_ID = "session1"

    async def main():
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )

        runner = Runner(
            agent=graph_rag_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        # Sample input
        payload = json.dumps(
            {
                "question": "For the brain regions Angular_R, Frontal_Sup_2_R, Parietal_Inf_R, Parietal_Sup_R, Frontal_Sup_Medial_R, Occipital_Mid_R, Postcentral_R, Temporal_Mid_R, Frontal_Mid_2_R, and SupraMarginal_R, what are their associated functional networks, cognitive functions, and known links to Alzheimer's Disease?"
            }
        )
        user_content = types.Content(role="user", parts=[types.Part(text=payload)])

        print("\n>>> Sending question to graph_rag_agent:\n", payload, "\n")

        # Run agent and collect final response
        final_result = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text

        print("\n<<< Final Agent Response:\n")
        print(final_result)

    asyncio.run(main())
