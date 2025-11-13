from google.adk.agents import LlmAgent
from pydantic import BaseModel

# 建議專案先 export GOOGLE_GENAI_USE_VERTEXAI=TRUE （或 config .env）

DEFAULT_MODEL = "gemini-2.5-flash-lite"

def create_llm_agent(
    name: str = None,
    description: str = None,
    model=DEFAULT_MODEL,
    instruction: str = None,
    output_schema: BaseModel = None,
    tools: list = None,
    output_key: str = None,
    disallow_transfer_to_peers: bool = True,
    disallow_transfer_to_parent: bool = True,
) -> LlmAgent:
    """
    建立 Vertex AI Gemini LLM Agent，支援 output_schema 及多工具。
    """
    agent_kwargs = {
        "name": name,
        "description": description,
        "model": model,
        "instruction": instruction,
        "tools": tools or [],
        "output_key": output_key,
        "disallow_transfer_to_peers": disallow_transfer_to_peers,
        "disallow_transfer_to_parent": disallow_transfer_to_parent,
    }
    if output_schema is not None:
        agent_kwargs["output_schema"] = output_schema

    return LlmAgent(**agent_kwargs)

# 使用 Pydantic schema 強制結構化輸出範例
class EmailContent(BaseModel):
    subject: str
    body: str

agent = create_llm_agent(
    name="email_gemini",
    description="用於電郵創建的 Gemini Agent",
    instruction="僅回傳 subject 和 body 字段的 JSON。",
    output_schema=EmailContent,
    output_key="email",
)

print(agent)