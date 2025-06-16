from google.adk import Agent
from .sub_agents.model_inspector.agent import model_inspector_agent

root_agent = Agent(
    name="fmri_explain_root",
    description="Root agent that coordinates the end-to-end fMRI explainability workflow.",
    instruction=(
        "You are the root controller responsible for managing all sub-agents "
        "in the fMRI explainability pipeline. Decide which sub-agent to call based on the user input."
    ),
    model="gemini-2.0-flash",  
    sub_agents=[model_inspector_agent],
)
