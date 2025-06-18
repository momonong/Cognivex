from google.adk import Agent
from .prompt import INSTRUCTION
from .tools import inspect_torch_model


model_inpector_agent = Agent(
    name="model_inspector_agent",
    model="gemini-2.0-flash",
    description="Agent that selects the most suitable layer from a PyTorch model for visualization or activation mapping.",
    instruction=INSTRUCTION,
    tools=[inspect_torch_model],
)
