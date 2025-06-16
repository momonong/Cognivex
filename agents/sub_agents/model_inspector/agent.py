from google.adk import Agent
from .prompt import INSTRUCTION
from .tools import extract_model_layers

model_inspector_agent = Agent(
    name="model_inspector_agent",
    model="gemini-2.0-flash",  # 可改為 "gemini-1.5-pro" 若需要
    description="Agent that selects the most suitable layer from a PyTorch model for visualization or activation mapping.",
    instruction=INSTRUCTION,
    tools=[extract_model_layers],
)
