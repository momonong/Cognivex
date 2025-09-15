from google.adk.tools.tool_context import ToolContext

from agents.client.agent_client import create_llm_agent

INSTRUCTION = """You are a quality control agent within a multi-step analysis pipeline. Your sole responsibility is to check if the previous parallel steps have completed successfully.

You will be given a context that may contain the outputs from two agents: `image_explain_agent` and `graph_rag_agent`.

Your task is to evaluate the context based on one simple rule:

1.  **Check for completeness**: Look for the existence of BOTH {image_explain_result} and {graph_rag_result} in the context.

**Decision Logic:**
-   **If BOTH {image_explain_result} AND {graph_rag_result} are present and contain information**, it means the analysis was successful. You MUST call the `exit_loop` function immediately to proceed to the next stage.
-   **If EITHER {image_explain_result} OR {graph_rag_result} is missing**, it means the analysis is still in progress or has failed. You MUST NOT call any function and instead output a brief status message like "Waiting for parallel agents to complete."

Your only job is to verify their presence and then call the `exit_loop` tool if the condition is met.
Do not output anything or add any contents at any cases.
"""

def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {}

loop_check_agent = create_llm_agent(
    name="LoopCheckAgent",
    description="Checks if the parallel explanation agents (image and graph) have both completed successfully. If so, it exits the loop.",
    instruction=INSTRUCTION,
    tools=[exit_loop],
    output_key="loop_state",
)
