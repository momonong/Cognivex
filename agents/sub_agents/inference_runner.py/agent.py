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

if __name__ == "__main__":
    # Example usage
    input_data = InferenceInput(
        model_path="model/capsnet/best_capsnet_rnn.pth",
        model_class="CapsNetRNN",
        model_module="scripts.capsnet.model",
        nii_path="data/subject_1.nii.gz",
        layer_name="conv1",
        window=5,
        stride=3,
        device="auto"
    )
    
    output = inference_agent.run(input_data)
    print(output)