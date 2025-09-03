import json
import asyncio
from google.adk.agents import LlmAgent

# Tools
from agents.sub_agents.graph_rag.tools.schema import summarize_graph_schema
from agents.sub_agents.graph_rag.tools.query import (
    graph_rag_query,
    regenerate_cypher_with_strategy,
)
from agents.sub_agents.graph_rag.tools.evaluate import evaluate_query

# -----------------------
# Define ADK Agent
# -----------------------

INSTRUCTIONS = """
You are a GraphRAG agent that answers graph-based questions using structured reasoning.

Conclude the question with {map_act_brain_result} to check the indo in knowlede graph.

Follow these steps strictly and do NOT skip any step:

1. First, call `summarize_graph_schema()` to understand the schema.
2. Then, call `graph_rag_query(question)` to get the Cypher query and its result.
3. Next, call `evaluate_query(question, result, ...)` to check if the result is valid and complete.
4. If `evaluate_query().is_valid == False`, then you MUST call `regenerate_cypher_with_strategy()` and re-run the query process starting from step 2.
5. **Once `evaluate_query()` returns `is_valid: True`, your task is complete. You must then synthesize the validated data and the original question into a final, natural language answer yourself.**

You MUST call `evaluate_query()` before generating the final answer.
Do NOT generate the final answer until the data is validated.
"""


graph_rag_agent = LlmAgent(
    name="GraphRAGAgent",
    model="gemini-2.5-flash",
    description="Agent for querying and reasoning over knowledge graphs using RAG.",
    instruction=INSTRUCTIONS,
    tools=[
        summarize_graph_schema,
        graph_rag_query,
        regenerate_cypher_with_strategy,
        evaluate_query,
    ],
    output_key="graph_rag_result",
)


# -----------------------
# Set up session & runner
# -----------------------


if __name__ == "__main__":
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
                "question": "Which three brain regions have the most functions associated with them in our knowledge graph?"
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