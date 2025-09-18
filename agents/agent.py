from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent
from google.adk.sessions import session

# Sub-agents
from agents.sub_agents.act_to_brain.agent import map_act_brain_agent
from agents.sub_agents.retrieve_img_path.agent import retrieve_img_path_agent
from agents.sub_agents.image_explain.agent import image_explain_agent
from agents.sub_agents.graph_rag.agent import graph_rag_agent
from agents.sub_agents.loop_manage.agent import loop_check_agent
from agents.sub_agents.final_report.agent import report_generator_agent


explain_parallel_agent = ParallelAgent(
    name="ExplainParallelAgent",
    description="A parallel agent that implement the image explain and graphrag at the same time.",
    sub_agents=[graph_rag_agent, image_explain_agent
    ],
)

# explain_loop_agent = LoopAgent(
#     name="ExplainLoopAgent",
#     description="A loop agent that make sure the parallel agent did get a result.",
#     sub_agents=[explain_parallel_agent, loop_check_agent],
# )

root_agent = SequentialAgent(
    name="fMRIAlzheimerPipeline",
    description="A multi-step neuroimaging analysis pipeline for Alzheimer's detection using deep learning and knowledge graph reasoning.",
    sub_agents=[
        map_act_brain_agent,
        retrieve_img_path_agent,
        explain_parallel_agent,
        report_generator_agent,
    ],
)


if __name__ == "__main__":
    import json
    import asyncio
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    APP_NAME = "fMRI_explain_pipeline"
    USER_ID = "test_user"
    SESSION_ID = "session-root-01"

    SUBJECT_ID = "sub-14"
    MODEL_PATH = "model/capsnet/best_capsnet_rnn.pth"
    NII_PATH = "data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz"

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

        payload = {
            "subject_id": SUBJECT_ID,
            "nii_path": NII_PATH,
            "model_path": MODEL_PATH,
        }

        user_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    text=f"Give me a thorough report of the subject with the following details:\n{json.dumps(payload, indent=2)}"
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
            print(f"--- Event Start ---")
            print(f"Agent Name: {getattr(event, 'agent_name', 'Unknown')}")
            print(f"Event Type: {type(event)}")
            print(f"Is Final Response: {event.is_final_response()}")
            if hasattr(event, "content") and event.content:
                print(
                    f"Content: {event.content.parts[0].text if event.content.parts else 'No Text Part'}"
                )
            print(f"--- Event End ---\n")

            # --- ★★★ 這裡新增 state 輸出 ★★★ ---


            session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
            print("All session state:", session.state)

            # print("EVENT_CONTENT:", event)
            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text

        print("\n<<< Final Agent Response:\n")
        if final_result:
            print(final_result)
        else:
            print("Agent returned no content.")

    asyncio.run(main())
