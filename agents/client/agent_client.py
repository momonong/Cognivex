from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel


# DEFAULT_MODEL = "gpt-oss-20b"
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def create_llm_agent(
    name: str = None,
    description: str = None,
    model = DEFAULT_MODEL,
    instruction: str = None,
    output_schema: BaseModel = None,
    tools: list = None,
    output_key: str = None,
) -> LlmAgent:

    if model.startswith("gpt-oss-20b"):
        agent = LlmAgent(
            name=name,
            description=description,
            model=LiteLlm(model="ollama_chat/gpt-oss:20b"),
            instruction=instruction,
            output_schema=output_schema,
            tools=tools or [],
            output_key=output_key,
            disallow_transfer_to_peers=True,
            disallow_transfer_to_parent=True,
        )

    if model.startswith("gemini-2.5-flash-lite"):
        if output_schema:
            agent = LlmAgent(
                name=name,
                description=description,
                model=model,
                instruction=instruction,
                output_schema=output_schema,
                output_key=output_key,
                disallow_transfer_to_peers=True,
                disallow_transfer_to_parent=True,
            )
        else:
            agent = LlmAgent(
                name=name,
                description=description,
                model=model,
                instruction=instruction,
                tools=tools,
                output_key=output_key,
            )

    return agent
