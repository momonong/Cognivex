import json
import asyncio
import time
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.agent import root_agent


async def run_analysis_async(subject_id: str, nii_path: str, model_path: str) -> str | None:
    """
    以非同步方式執行 fMRI 分析的 Agent Pipeline。

    Args:
        subject_id (str): 受試者 ID。
        nii_path (str): NIfTI 檔案的路徑。
        model_path (str): 模型檔案 (.pth) 的路徑。

    Returns:
        str | None: Agent pipeline 回傳的最終結果字串 (通常是 JSON)，如果沒有結果則回傳 None。
    """
    # 1. 填寫常數並動態生成 SESSION_ID
    APP_NAME = "fMRI_Explain_Pipeline"
    USER_ID = "test_user"
    # 每次執行都建立一個獨立的 session，避免狀態互相干擾
    SESSION_ID = f"session-{subject_id}-{int(time.time())}"

    # 2. 建立 ADK Session 和 Runner
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

    # 3. 準備傳送給 Agent 的 payload
    payload = {
        "subject_id": subject_id,
        "nii_path": nii_path,
        "model_path": model_path,
    }

    user_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=f"Give me a thorough report of the subject with the following details:\n{json.dumps(payload, indent=2)}"
            )
        ],
    )

    print(f"\n>>> 正在發送請求至 Agent (Session: {SESSION_ID})...\n")

    # 4. 執行 Agent Pipeline 並取得結果
    final_result = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_result = event.content.parts[0].text

    print("\n<<< Agent Pipeline 最終回傳:\n")
    if final_result:
        print(final_result)
    else:
        print("Agent 沒有回傳任何內容。")
    
    # 5. 回傳結果
    return final_result


def run_analysis_sync(subject_id: str, nii_path: str, model_path: str) -> str | None:
    """
    一個同步的包裝函式，讓 Streamlit 或其他同步程式可以輕易呼叫。
    """
    return asyncio.run(run_analysis_async(subject_id, nii_path, model_path))


# --- 範例執行區塊 ---
if __name__ == "__main__":
    # 模擬從前端傳入的參數，用於獨立測試
    test_subject_id = "sub-14"
    test_nii_path = "data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz"
    test_model_path = "model/capsnet/best_capsnet_rnn.pth"
    
    print("="*50)
    print("正在以獨立模式執行後端分析腳本...")
    print("="*50)

    # 呼叫同步函式來執行整個流程
    result = run_analysis_sync(test_subject_id, test_nii_path, test_model_path)

    if result:
        print("\n--- 測試執行成功 ---")
        # 嘗試解析看看回傳的是否為有效的 JSON
        try:
            report_data = json.loads(result)
            print("回傳的 JSON 解析成功！")
        except json.JSONDecodeError:
            print("回傳的內容不是有效的 JSON 格式。")
    else:
        print("\n--- 測試執行完成，但沒有收到回傳結果 ---")