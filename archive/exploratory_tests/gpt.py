# import os
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# completion = client.chat.completions.create(
#     model="openai/gpt-oss-20b",
#     messages=[
#       {
#         "role": "user",
#         "content": "Can you help me explain a fMRI image?",
#       }
#     ],
#     temperature=1,
#     max_completion_tokens=8192,
#     top_p=1,
#     reasoning_effort="medium",
#     stream=True,
#     stop=None
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")
#     

# from google.adk.agents import LlmAgent
# from google.adk.models.lite_llm import LiteLlm
# # Import the complete pipeline tool (now includes save_results)
# from agents.sub_agents.act_to_brain.tools.pipeline import pipeline


# agent_openai = LlmAgent(
#     model=LiteLlm(model="ollama_chat/gpt-oss:20b"), # LiteLLM model string format
#     name="openai_agent",
#     instruction="You are a helpful assistant powered by gpt-oss-20b.",
# )


# if __name__ == "__main__":
#     import asyncio
#     from google.adk.runners import Runner
#     from google.adk.sessions import InMemorySessionService
#     from google.genai import types

#     APP_NAME = "test_agent_pipeline"
#     USER_ID = "neuroimaging_user"
#     SESSION_ID = "session-001"

#     async def main():
#         """
#         Run the NIfTI inference agent with proper ADK session management.
#         """
#         # Create session service
#         session_service = InMemorySessionService()
#         await session_service.create_session(
#             app_name=APP_NAME, 
#             user_id=USER_ID, 
#             session_id=SESSION_ID
#         )

#         # Create runner
#         runner = Runner(
#             agent=agent_openai,
#             app_name=APP_NAME,
#             session_service=session_service,
#         )

#         # Prepare user message
#         user_message = types.Content(
#             role="user",
#             parts=[types.Part(
#                 text="Can you help me explain a fMRI image?"
#             )],
#         )

#         print("\n" + "="*80)
#         print("STARTING NII INFERENCE AGENT (with save_results integration)")
#         print("="*80)
#         print(f"App: {APP_NAME}")
#         print(f"User: {USER_ID}")
#         print(f"Session: {SESSION_ID}")
#         print(f"Agent: {agent_openai.name}")
#         print("\n>>> Sending request to map_act_brain_agent...\n")

#         # Run the agent
#         final_result = None
#         async for event in runner.run_async(
#             user_id=USER_ID,
#             session_id=SESSION_ID,
#             new_message=user_message,
#         ):
#             if event.is_final_response() and event.content and event.content.parts:
#                 final_result = event.content.parts[0].text
                
#         print("\n" + "="*80)
#         print("ACT TO BRAIN AGENT RESPONSE")
#         print("="*80)
#         print(final_result)
#         print("\n" + "="*80)
        
#         return final_result

#     # Run the async main function
#     result = asyncio.run(main())
