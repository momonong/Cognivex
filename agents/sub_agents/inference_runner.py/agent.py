from google.adk import Agent
from .prompt import INSTRUCTION  # 可寫推論與 activation 擷取說明
from .tools import run_inference_and_save_activation
from .interface import InferenceInput, InferenceOutput

inference_agent = Agent(
    name="gemini-2.0-flash-lite",
    model="gemini-1.5-flash",
    instruction=INSTRUCTION,
    input_type=InferenceInput,
    output_type=InferenceOutput,
    function=run_inference_and_save_activation,
)
