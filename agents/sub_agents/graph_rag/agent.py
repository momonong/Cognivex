import json
import asyncio

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Tools
from agents.sub_agents.graph_rag.tools.schema import summarize_graph_schema
from agents.sub_agents.graph_rag.tools.query import (
    graph_rag_query,
    regenerate_cypher_with_strategy,
)
from agents.sub_agents.graph_rag.tools.evaluate import evaluate_query

# -----------------------
# 1. Define ADK Agent
# -----------------------

INSTRUCTIONS = """
You are a GraphRAG agent that answers graph-based questions using structured reasoning.

Follow these steps strictly and do NOT skip any step:

1. First, call `summarize_graph_schema()` to understand the schema.
2. Then, call `graph_rag_query(question)` to get the Cypher query and its result.
3. Next, call `evaluate_query(question, result, expected_fields=["region", "function"], min_count=2)` to check if the result is valid and complete. You MUST run this step even if the result looks fine.
4. If `evaluate_query().is_valid == False`, then you MUST call `regenerate_cypher_with_strategy()` using the provided strategy and re-run the Cypher.
5. Once the result is valid and complete, call `synthesize_answer()` to generate the final answer.
6. Finally, return: { "answer": "<your final natural language answer>" }

You MUST call `evaluate_query()` before deciding whether to answer or retry.
Do NOT skip evaluation even if the query result looks complete.

NEVER return the final answer before evaluation.

Example format:
{
  "answer": "The regions related to Alzheimer's disease include A, B, and C. These are associated with functions such as X, Y, and Z."
}
"""


graph_rag_agent = Agent(
    name="graph_rag_agent",
    model="gemini-2.5-flash",
    description="Agent for querying and reasoning over knowledge graphs using RAG.",
    instruction=INSTRUCTIONS,
    tools=[
        summarize_graph_schema,
        graph_rag_query,
        regenerate_cypher_with_strategy,
        evaluate_query,
    ],
    output_key="answer",
)

# -----------------------
# 2. Set up session & runner
# -----------------------

APP_NAME = "graph_rag_adk"
USER_ID = "test_user"
SESSION_ID = "session1"

# -----------------------
# 3. Test Driver (CLI entrypoint)
# -----------------------


if __name__ == "__main__":

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
                "question": "What brain regions are associated with Alzheimer's disease, and what functions do they perform?"
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

        # Optional: show session memory
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        print("Session state:", json.dumps(session.state, indent=2))

    asyncio.run(main())
