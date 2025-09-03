from google.adk.agents import SequentialAgent, LlmAgent

# Sub-agents
from agents.sub_agents.act_to_brain.agent import map_act_brain_agent
from agents.sub_agents.image_explain.agent import image_explain_agent
from agents.sub_agents.graph_rag.agent import graph_rag_agent
from agents.sub_agents.final_report.agent import report_generator_agent

INSTRUCTION = """
You need to strictly follow the order and run all pipelines:
1. Run the `map_act_brain_agent` to process the fMRI data and extract activation patterns.
2. Use the `image_explain_agent` to generate a clinical explanation of the activation maps.
3. Execute the `graph_rag_agent` to query the knowledge graph for additional insights.
4. Finally, call the `report_generator_agent` to compile all results into a comprehensive clinical report.
"""
root_agent = SequentialAgent(
    name="fMRI_Alzheimer_Pipeline",
    description="A multi-step neuroimaging analysis pipeline for Alzheimer's detection using deep learning and knowledge graph reasoning.",
    sub_agents=[
        map_act_brain_agent,
        image_explain_agent,
        graph_rag_agent,
        report_generator_agent,
    ],
)


if __name__ == "__main__":
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    APP_NAME = "fMRI_explain_pipeline"
    USER_ID = "test_user"
    SESSION_ID = "session-root-01"

    async def main():
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )

        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        user_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    text="Give me a thorough report of the subject. Subject ID: 'sub-14'."
                )
            ],
        )

        print("\n>>> Sending request to root agent...\n")

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
