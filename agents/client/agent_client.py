import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

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
    disallow_transfer_to_peers: bool = True,
    disallow_transfer_to_parent: bool = True,
) -> LlmAgent:
    """
    Creates a Vertex AI Gemini or GPT OSS LLM Agent.
    Supports output_schema, tools, and agent state transfer control.
    """
    agent_kwargs = {
        "name": name,
        "description": description,
        "instruction": instruction,
        "tools": tools or [],
        "output_key": output_key,
        "disallow_transfer_to_peers": disallow_transfer_to_peers,
        "disallow_transfer_to_parent": disallow_transfer_to_parent,
    }

    # GPT OSS agent: uses LiteLlm for the model
    if model.startswith("gpt-oss-20b"):
        agent_kwargs["model"] = LiteLlm(model="ollama_chat/gpt-oss:20b")
        if output_schema is not None:
            agent_kwargs["output_schema"] = output_schema
        return LlmAgent(**agent_kwargs)

    # Gemini Agent (suitable for Vertex AI trial credit): uses the model name directly
    if model.startswith("gemini-2.5-flash-lite"):
        agent_kwargs["model"] = model
        if output_schema is not None:
            agent_kwargs["output_schema"] = output_schema
        return LlmAgent(**agent_kwargs)

    # Other models can be extended with additional branches
    
    # Returns None by default (or custom exception handling)
    return None
