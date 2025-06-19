from google.adk import Agent
from .prompt import INSTRUCTION
from .tools import extract_model_layers

def create_model_inspector_agent(model_path: str):
    # Bind model_path to the tool function
    tools = [lambda: extract_model_layers(model_path)]
    
    return Agent(
        name="model_inspector_agent",
        model="gemini-2.0-flash",
        description="Agent that selects the most suitable layer from a PyTorch model for visualization or activation mapping.",
        instruction=INSTRUCTION,
        tools=tools,
    )

if __name__ == "__main__":
    # User-side code
    model_path = "/path/to/your_model.pth"
    agent = create_model_inspector_agent(model_path)

    # Execute the tool
    layer_list = agent.tools[0]()
    print(layer_list)  # 輸出模型層列表
