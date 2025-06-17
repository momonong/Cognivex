from google.adk import Agent
from .prompt import INSTRUCTION
from .tools import extract_model_layers

def create_model_inspector_agent(model_path: str):
    # 將 model_path 綁定到工具函數
    tools = [lambda: extract_model_layers(model_path)]
    
    return Agent(
        name="model_inspector_agent",
        model="gemini-2.0-flash",
        description="Agent that selects the most suitable layer from a PyTorch model for visualization or activation mapping.",
        instruction=INSTRUCTION,
        tools=tools,
    )
