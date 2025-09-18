import json
import asyncio
import time
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.agent import root_agent


async def run_analysis_async(subject_id: str, nii_path: str, model_path: str) -> str | None:
    """
    Execute fMRI analysis Agent Pipeline asynchronously.

    Args:
        subject_id (str): Subject ID.
        nii_path (str): Path to NIfTI file.
        model_path (str): Path to model file (.pth).

    Returns:
        str | None: Final result string returned by Agent pipeline (usually JSON), None if no result.
    """
    # 1. Fill in constants and dynamically generate SESSION_ID
    APP_NAME = "fMRI_Explain_Pipeline"
    USER_ID = "test_user"
    # Create an independent session for each execution to avoid state interference
    SESSION_ID = f"session-{subject_id}-{int(time.time())}"

    # 2. Create ADK Session and Runner
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

    # 3. Prepare payload to send to Agent
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

    print(f"\n>>> Sending request to Agent (Session: {SESSION_ID})...\n")

    # 4. Execute Agent Pipeline and get results
    final_result = None
    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=user_message,
        ):
            session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
            print("All session state:", session.state)

            if event.is_final_response() and event.content and event.content.parts:
                final_result = event.content.parts[0].text
    except Exception as e:
        # This block now only catches critical failures in the ADK runner itself,
        # not pipeline logic errors, which are handled by the LoopManagerAgent.
        print(f"A critical error occurred in the agent runner: {e}")

    print("\n<<< Agent Pipeline final response:\n")
    if final_result:
        print(final_result)
    else:
        print("Agent returned no content.")
    
    # 5. Return results
    return final_result


def run_analysis_sync(subject_id: str, nii_path: str, model_path: str) -> str | None:
    """
    A synchronous wrapper function that allows Streamlit or other synchronous programs to easily call.
    """
    return asyncio.run(run_analysis_async(subject_id, nii_path, model_path))


# --- Example execution block ---
if __name__ == "__main__":
    # Simulate parameters passed from frontend for independent testing
    test_subject_id = "sub-14"
    test_nii_path = "data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz"
    test_model_path = "model/capsnet/best_capsnet_rnn.pth"
    
    print("="*50)
    print("Running backend analysis script in standalone mode...")
    print("="*50)

    # Call synchronous function to execute the entire process
    result = run_analysis_sync(test_subject_id, test_nii_path, test_model_path)

    if result:
        print("\n--- Test execution successful ---")
        # Try to parse if the returned content is valid JSON
        try:
            report_data = json.loads(result)
            print("Returned JSON parsed successfully!")
        except json.JSONDecodeError:
            print("Returned content is not in valid JSON format.")
    else:
        print("\n--- Test execution completed but no return result received ---")
