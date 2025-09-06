from google.adk.agents import SequentialAgent, ParallelAgent

# Sub-agents
from agents.sub_agents.act_to_brain.agent import map_act_brain_agent
from agents.sub_agents.image_explain.agent import image_explain_agent
from agents.sub_agents.graph_rag.agent import graph_rag_agent
from agents.sub_agents.final_report.agent import report_generator_agent


explain_parallel_agent = ParallelAgent(
    name="ExplainParallelAgent", sub_agents=[image_explain_agent, graph_rag_agent]
)

root_agent = SequentialAgent(
    name="fMRIAlzheimerPipeline",
    description="A multi-step neuroimaging analysis pipeline for Alzheimer's detection using deep learning and knowledge graph reasoning.",
    sub_agents=[
        map_act_brain_agent,
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
                    text="Give me a thorough report of the subject with the following details:\n"
                        f"{json.dumps(payload, indent=2)}"
                )
            ],
        )

        print("\n>>> Sending request to root agent...\n")

        # --- 步驟 1: 初始化成果容器 ---
        all_agent_outputs = {}
        final_text_report = None

        # --- 步驟 2: 升級事件處理迴圈 ---
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message,
        ):
            # 監聽每一個子 Agent 的結束事件
            if event.is_agent_end() and event.actions and event.actions.state_delta:
                agent_name = event.author
                # 提取該 Agent 新增到 state 中的結構化數據
                structured_output = event.actions.state_delta
                
                print("\n" + "-"*30)
                print(f"✅ CAPTURED OUTPUT from: {agent_name}")
                print(json.dumps(structured_output, indent=2))
                print("-"*30)
                
                # 將結果儲存到我們的容器中
                all_agent_outputs[agent_name] = structured_output

            # 仍然可以保留對最終文字報告的捕捉
            if event.is_final_response() and event.content and event.content.parts:
                final_text_report = event.content.parts[0].text

        # --- 步驟 3: 在迴圈後，可以對收集到的所有結構化成果進行處理 ---
        print("\n" + "="*80)
        print("  FINAL DISTILLED RESULTS (from all agents)")
        print("="*80)
        
        # 美化打印所有收集到的成果
        print(json.dumps(all_agent_outputs, indent=2))

        print("\n" + "="*80)
        print("  FINAL NARRATIVE REPORT")
        print("="*80)
        print(final_text_report)

    # 運行主程式
    asyncio.run(main())
