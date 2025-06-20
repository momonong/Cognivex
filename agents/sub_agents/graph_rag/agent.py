import json
import asyncio

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import tools
from agents.sub_agents.graph_rag.tools import schema, query, evaluate

# 1. Define the Agent with its tools
INSTRUCTIONS = """
You are a GraphRAG agent that uses tools to answer graph questions.
Workflow:
1. Call summarize_graph_schema() → get schema summary.
2. Call generate_cypher() with the question → get Cypher.
3. Call run_query() with the Cypher → get raw results.
4. Call evaluate_and_decide() with cypher, question & results → decide if query is good.
5. If evaluation fails, you may re-ask or adjust externally.
6. Return a JSON matching { "answer": "<your answer>" }.
"""

graph_rag_agent = Agent(
    name="graph_rag_agent",
    model="gemini-2.5-flash",
    description="Agent for querying and reasoning over knowledge graphs using RAG.",
    instruction=INSTRUCTIONS,
    tools=[
        schema.summarize_graph_schema,
        query.graph_rag_answer,
        # evaluate.evaluate_and_decide,
    ],
    input_schema=None,  # we’ll pass raw JSON
    output_schema=None,  # agent returns free-form JSON with "answer"
    output_key="answer",
)

# 2. Set up session & runner
APP_NAME = "graph_rag_adk"
USER_ID = "test_user"
SESSION_ID = "session1"


# 3. Test driver
async def main():
    # 初始化 session service 與 runner
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=graph_rag_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # 測試問題
    payload = json.dumps(
        {
            "question": "What brain regions are associated with Alzheimer's disease, and what functions do they perform?",
        }
    )
    user_content = types.Content(role="user", parts=[types.Part(text=payload)])

    print("\n>>> Sending question to graph_rag_agent:\n", payload, "\n")

    final_answer = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_answer = event.content.parts[0].text

    print("\n<<< Agent’s final response:\n", final_answer, "\n")

    # Optional: 檢查 session 狀態
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    print("Session state:", json.dumps(session.state, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
