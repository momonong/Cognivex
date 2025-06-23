from google.adk.agents import SequentialAgent

# Sub-agents
from agents.sub_agents.nii_inference.agent import nii_inference_agent
from agents.sub_agents.image_explain.agent import image_explain_agent
from agents.sub_agents.graph_rag.agent import graph_rag_agent
from agents.sub_agents.final_report.agent import report_generator_agent


root_agent = SequentialAgent(
    name="fMRI_Alzheimer_Pipeline",
    description="A multi-step neuroimaging analysis pipeline for Alzheimer's detection using deep learning and knowledge graph reasoning.",
    sub_agents=[
        nii_inference_agent,
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
            parts=[types.Part(text="Analyze fMRI subject using the pipeline.")],
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
